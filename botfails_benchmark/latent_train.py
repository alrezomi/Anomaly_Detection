"""Joint BotFails LoRA + logistic feature-head adaptation on development bags only.

The base/vision weights stay frozen. BCE directly supervises the same final
prompt vector used at inference; no execution answer text is generated.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .data import TASKS, episodes, load_image
from .latent_adapt import causal_context_frames, dump, labels


def sample_at(episode, frames, truth, requested):
    available = [f for f in frames if f[1] >= 1.]
    if not available:
        raise ValueError('No camera observations after one second: ' + episode.key)
    frame = next((f for f in available if f[1] >= requested), available[-1])
    onset = float(np.argmax(truth)/30) if truth is not None and truth.any() else None
    return dict(bag=episode.key, task=episode.task, split=episode.split, episode=episode.name,
                frame_index=frame[0], timestamp_sec=frame[1],
                target=int(onset is not None and frame[1] >= onset),bag_failure=onset is not None)


def epoch_samples(episode, frames, truth, epoch, index):
    """Paired before/after faults; normal progress varies deterministically by epoch."""
    end = frames[-1][1]
    if truth is not None and truth.any():
        onset = float(np.argmax(truth)/30)
        requested = [max(1., onset-.75), min(end, onset+1.) if epoch % 2 else end]
    else:
        requested = [(1., (1.+end)/2, end)[(index+epoch-1) % 3]]
    result = {s['frame_index']:s for s in [sample_at(episode,frames,truth,t) for t in requested]}
    return list(result.values())


def adapter_digest(folder, profile):
    digest = hashlib.sha256()
    for name in ('adapter_config.json','adapter_model.safetensors'):
        digest.update((folder/name).read_bytes())
    digest.update(json.dumps(profile,sort_keys=True).encode())
    return digest.hexdigest()


def paired_order(samples, seed, rounds=4):
    """Equal task/class contribution in every update, including after gradient clipping."""
    rng=random.Random(seed)
    tasks=sorted({s['task'] for s in samples});result=[];pools={}
    for task in tasks:
        positives={}
        negatives=[s for s in samples if s['task']==task and not s['target']]
        for sample in samples:
            if sample['task']==task and sample['target']:
                positives.setdefault(sample['bag'],[]).append(sample)
        if not positives or not negatives:raise ValueError('Both prefix classes are required for task '+task)
        keys=list(positives);rng.shuffle(keys)
        pools[task]=(positives,keys,negatives)
    for iteration in range(rounds):
        rng.shuffle(tasks)
        for task in tasks:
            positives,keys,negatives=pools[task]
            positive=rng.choice(positives[keys[iteration % len(keys)]])
            matched=[s for s in negatives if s['bag']==positive['bag']]
            normal=[s for s in negatives if not s.get('bag_failure',False)]
            pool=matched if iteration%2==0 and matched else normal or negatives
            pair=[rng.choice(pool),positive];rng.shuffle(pair);result.extend(pair)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--no-gradient-checkpointing',action='store_true')
    parser.add_argument('--balanced-pairs',action='store_true')
    parser.add_argument('--initialize-adapter',type=Path)
    args = parser.parse_args()
    import torch
    from torch.nn import functional as F
    from pipeline_settings import load_pipeline_config
    from rynnbrain_vlm.model import RynnBrainModel,COP_REPRESENTATION_ID
    from rynnbrain_vlm.lora import attach_lora
    from rynnbrain_vlm.cop_classifier import fit_logistic_classifier,CoPLogisticClassifier,save_classifier
    if (args.output/'training_status.json').exists():
        raise FileExistsError('Choose a new output for a new joint training run.')
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/'training_status.json',dict(state='preparing'))
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    free,total = torch.cuda.mem_get_info()
    cap = min(40*1024**3, int(.7*free))
    if cap < 24*1024**3:
        raise RuntimeError('Insufficient free GPU memory for the bounded training pilot.')
    torch.cuda.set_per_process_memory_fraction(cap/total)
    print('Training allocator cap GiB',cap/1024**3,flush=True)
    config = load_pipeline_config(args.config)['rynnbrain']
    template_signature = json.loads(Path(config['cop_classifier']['model_paths']['raw']).with_suffix('.json').read_text())['comparison_signature']
    model_config = {**config['model'],'lora_adapter_path':str(args.initialize_adapter) if args.initialize_adapter else None,
                    'dtype':'bfloat16','device_map':{'':'cuda:0'}}
    wrapper = RynnBrainModel(model_config)
    norm = wrapper._final_language_norm()
    refs = {t:json.loads((args.references/(t+'.json')).read_text()) for t in TASKS}
    topic = '/top_camera/cam_135222070433/color/image_raw/compressed'
    def labelled(frame,k):
        return (f'Frame {k+1}, camera {topic}, time {frame[1]:.3f} s from this bag start:',load_image(frame[2]))
    reference_turns = {t:dict(role='user',text=r['turn1'],images=[labelled((0,int(Path(path).stem)/30,Path(path)),k)
                       for k,path in enumerate(r['images'])]) for t,r in refs.items()}
    bags,tables,truths = {},{},{}
    fitting,validation = [],[]
    def register(ep,group):
        bags[ep.key]=ep;tables[ep.key]=ep.frame_table(refs[ep.task]['camera']);truths[ep.key]=labels(ep)
        group.append(ep.key)
    for task in TASKS:
        normal = episodes(args.dataset,args.frames,'normal_train',task)
        test = episodes(args.dataset,args.frames,'test',task)
        for ep in normal[1:5]+[e for e in test if e.index % 5==0]:register(ep,fitting)
        development = [e for e in test if e.index % 5==1]
        val_normal = next((e for e in development if not labels(e).any()),normal[5])
        val_failure = next(e for e in development if labels(e).any())
        register(val_normal,validation);register(val_failure,validation)
    assert not set(fitting)&set(validation)
    assert all(bags[k].split=='normal_train' or bags[k].index % 5==0 for k in fitting)
    epochs = [[s for index,key in enumerate(fitting) for s in epoch_samples(bags[key],tables[key],truths[key],epoch,index)]
              for epoch in range(1,args.epochs+1)]
    val_samples = []
    for key in validation:
        val_samples.append(sample_at(bags[key],tables[key],truths[key],tables[key][-1][1]))
        if truths[key] is not None and truths[key].any():
            val_samples.append(sample_at(bags[key],tables[key],truths[key],max(1.,np.argmax(truths[key])/30-.75)))
    dump(args.output/'data_split.json',dict(fitting_bags=fitting,validation_bags=validation,epochs=epochs,validation_samples=val_samples,
        reserved_test_indices_modulo_five=[2,3,4],reference_bags={t:r['reference'] for t,r in refs.items()}))
    (args.output/'trainer_snapshot.py').write_text(Path(__file__).read_text())
    state = {}
    def capture(_module,_inputs,output):
        h = output[0] if isinstance(output,tuple) else output
        if isinstance(h,torch.Tensor) and h.ndim==3:state['hidden']=h[0,-1]
    handle = norm.register_forward_hook(capture)
    def encode(sample):
        task,key = sample['task'],sample['bag']
        current = dict(role='user',text=refs[task]['turn2'],images=[labelled(f,k) for k,f in enumerate(
                       causal_context_frames(tables[key],sample['timestamp_sec']))])
        conversation = [wrapper.conversation_message(reference_turns[task]),
            dict(role='assistant',content=[dict(type='text',text=refs[task]['response'])]),wrapper.conversation_message(current)]
        inputs = wrapper.tokenize_conversation(conversation,config['generation']).to(wrapper.input_device)
        state.clear()
        wrapper.model(**inputs,use_cache=False,logits_to_keep=1)
        hidden = state.pop('hidden').float()
        assert hidden.shape==(4096,) and torch.isfinite(hidden).all()
        return hidden
    head = torch.nn.Linear(4096,1,device='cuda',dtype=torch.float32)
    if args.initialize_adapter:
        from rynnbrain_vlm.cop_classifier import load_classifier
        initial_head=load_classifier(args.initialize_adapter/'classifier.npz')
        assert initial_head.comparison_signature['lora_adapter_sha256']==wrapper.adapter_identity
        assert initial_head.comparison_signature['reference_contexts']==refs
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(initial_head.weights[None],device='cuda',dtype=torch.float32))
            head.bias.fill_(initial_head.intercept-float(initial_head.feature_mean@initial_head.weights))
    elif not args.smoke_steps:
        vectors,targets=[],[]
        wrapper.model.eval()
        with torch.inference_mode():
            for index,key in enumerate(fitting):
                sample=sample_at(bags[key],tables[key],truths[key],tables[key][-1][1])
                vectors.append(encode(sample).cpu().numpy());targets.append(sample['target'])
                if (index+1)%10==0:print('Head initialization features',index+1,len(fitting),flush=True)
        weights,intercept,mean,diagnostics=fit_logistic_classifier(vectors,targets,c_value=100.)
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(weights[None],device='cuda',dtype=torch.float32))
            head.bias.fill_(intercept-float(mean@weights))
        dump(args.output/'head_initialization.json',dict(prefixes=len(targets),diagnostics=diagnostics))
    if args.initialize_adapter:
        for name,parameter in wrapper.model.named_parameters():
            parameter.requires_grad_('lora_' in name and '.language_model.' in name)
        wrapper.model.peft_config['default'].inference_mode=False
    else:
        wrapper.model = attach_lora(wrapper.model,dict(rank=8,alpha=16,dropout=.05,target_modules=['q_proj','v_proj']))
    model=wrapper.model
    model.config.use_cache=False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    lora_parameters=[p for p in model.parameters() if p.requires_grad]
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n and '.language_model.' in n for n in trainable)
    optimizer=torch.optim.AdamW([dict(params=lora_parameters,lr=5e-5),dict(params=head.parameters(),lr=1e-3)],weight_decay=0.)
    watched=next(p for n,p in model.named_parameters() if 'lora_B' in n)
    initial=watched.detach().clone()
    profile=dict(protocol='timestamped_multiview_prefix_decision_onset_v1',camera_topics=[topic],memory_camera_topics=[topic],
                 num_frames=8,input_mode='raw',source='BotFails_frames',timestamp_decimals=3,context_sampling='causal_uniform')
    manifest=dict(base_model=model_config['model_id'],model_config=model_config,temporal_profile=profile,
        objective='BCE on L2-normalized final language prompt state; train only language LoRA and one global logistic head.',
        input_mode='raw',representation_id=COP_REPRESENTATION_ID,reference_contexts=refs,
        trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=42,learning_rate=5e-5,head_learning_rate=1e-3,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        balanced_same_task_pairs=args.balanced_pairs,initial_adapter_sha256=wrapper.adapter_identity,
        bags=[dict(bag_name='BotFails__'+key.replace('/','__'),bag_path=key,split='train' if key in fitting else 'validation',
                   label='fail' if truths[key] is not None and truths[key].any() else 'normal') for key in fitting+validation])
    def save(folder,step):
        folder.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(folder,safe_serialization=True)
        dump(folder/'training_manifest.json',{**manifest,'optimizer_step':step})
        identity=adapter_digest(folder,profile)
        signature=dict(model_id=model_config['model_id'],lora_adapter_sha256=identity,representation_id=COP_REPRESENTATION_ID,
            hidden_size=4096,input_mode='raw',context_sampling='causal_uniform',num_frames=8,model_config={**model_config,'lora_adapter_path':str(folder)},
            reference_contexts=refs,temporal_profile=profile,enable_thinking=bool(config['generation'].get('enable_thinking',False)),
            task_description='__BOTFAILS_TASK__',test_camera_topics=[topic],memory_camera_topics=[topic],
            turn1_prompt=template_signature['turn1_prompt'].replace(template_signature['task_description'],'__BOTFAILS_TASK__'),
            turn2_prompt=template_signature['turn2_prompt'].replace(template_signature['task_description'],'__BOTFAILS_TASK__'))
        metadata=dict(schema_version=1,classifier_type='timestamp_supervised_joint_logistic_regression',vector_dimension=4096,
            input_mode='raw',representation_id=COP_REPRESENTATION_ID,comparison_signature=signature,threshold=.5,
            training_bags=fitting,validation_bags=validation,training_options=manifest,optimizer_step=step)
        classifier=CoPLogisticClassifier(head.weight.detach().float().cpu().numpy()[0],float(head.bias.detach()),
            np.zeros(4096),.5,'raw',COP_REPRESENTATION_ID,signature,metadata)
        save_classifier(folder/'classifier.npz',classifier)
        return identity
    last_validation_vectors=[]
    def validate():
        model.eval();head.eval();losses={0:[],1:[]};correct={0:[],1:[]};predictions=[]
        last_validation_vectors.clear()
        with torch.no_grad():
            for sample in val_samples:
                vector=encode(sample)
                last_validation_vectors.append(vector.cpu().numpy())
                z=head(F.normalize(vector,dim=0)).reshape(())
                target=torch.tensor(float(sample['target']),device='cuda')
                losses[sample['target']].append(float(F.binary_cross_entropy_with_logits(z,target)))
                correct[sample['target']].append(bool((z>=0).item())==bool(sample['target']))
                predictions.append({**sample,'raw_logit':float(z),'score':float(torch.sigmoid(z))})
        from sklearn.metrics import roc_auc_score
        y=np.asarray([s['target'] for s in predictions],bool)
        scores=np.asarray([s['score'] for s in predictions])
        thresholds=np.r_[-np.inf,np.unique(scores),np.inf]
        recall_at_90=max(float((scores[y]>=t).mean()) for t in thresholds if float((scores[~y]<t).mean())>=.9)
        return dict(loss=float(np.mean([np.mean(v) for v in losses.values() if v])),
                    prefix_TPR=float(np.mean(correct[1])),prefix_TNR=float(np.mean(correct[0])),
                    AUC=float(roc_auc_score(y,scores)),TPR_at_90_TNR=recall_at_90,prefix_predictions=predictions)
    best_loss=float('inf');best_quality=(-1.,-1.,-float('inf'));best_step=0;step=0;stale=0;history=[]
    if args.smoke_steps:
        positive=next(s for s in epochs[0] if s['target'])
        negative=next(s for s in epochs[0] if s['bag']==positive['bag'] and not s['target'])
        epochs=[[negative,positive]*args.smoke_steps]
    parameters=lora_parameters+list(head.parameters())
    def select(result,step):
        nonlocal best_loss,best_quality,best_step,stale
        quality=(result['TPR_at_90_TNR'],result['AUC'],-result['loss'])
        if quality>best_quality:
            best_quality=quality;best_loss=result['loss'];best_step=step;stale=0
            identity=save(args.output/'adapter_best',step)
            np.savez_compressed(args.output/'adapter_best/validation_vectors.npz',vectors=np.stack(last_validation_vectors))
            dump(args.output/'adapter_best/validation_features.json',dict(adapter_sha256=identity,samples=val_samples))
            dump(args.output/'best.json',dict(step=step,adapter_sha256=identity,**result))
        else:stale+=1
    if args.initialize_adapter and not args.smoke_steps:
        initial_result=validate();select(initial_result,0)
        print('INITIAL VALIDATION',{k:v for k,v in initial_result.items() if k!='prefix_predictions'},flush=True)
    for epoch,order in enumerate(epochs,1):
        if args.balanced_pairs and not args.smoke_steps:
            order=paired_order(order,42+epoch)
        else:
            random.Random(42+epoch).shuffle(order)
        counts=Counter((s['task'],s['target']) for s in order);tasks=len({s['task'] for s in order})
        for start in range(0,len(order),2):
            model.train();model.get_base_model().model.visual.eval();head.train();optimizer.zero_grad(set_to_none=True)
            group=order[start:start+2];loss_value=0.;started=time.perf_counter()
            for sample in group:
                hidden=encode(sample)
                assert hidden.requires_grad,'The training readout lost its gradient.'
                z=head(F.normalize(hidden,dim=0)).reshape(())
                target=torch.tensor(float(sample['target']),device='cuda')
                weight=len(order)/(2*tasks*counts[sample['task'],sample['target']])
                loss=F.binary_cross_entropy_with_logits(z,target)*weight
                if not torch.isfinite(loss):raise RuntimeError('Non-finite classification loss.')
                loss_value+=float(loss.detach());(loss/len(group)).backward()
            gradient=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            optimizer.step();step+=1
            item=dict(epoch=epoch,step=step,loss=loss_value/len(group),gradient_norm=gradient,
                      elapsed_sec=time.perf_counter()-started,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
            history.append(item);dump(args.output/'progress.json',item)
            with (args.output/'history.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            print('TRAIN',item,flush=True)
            if args.smoke_steps and step>=args.smoke_steps:
                delta=float(torch.linalg.vector_norm(watched.detach()-initial))
                assert delta>0,'No LoRA weight changed.'
                model.eval();head.eval()
                with torch.no_grad():
                    vector=encode(positive).cpu().numpy()
                    probability=float(torch.sigmoid(head(F.normalize(torch.as_tensor(vector,device='cuda'),dim=0))))
                identity=save(args.output/'smoke_adapter',step)
                from rynnbrain_vlm.cop_classifier import load_classifier
                saved=load_classifier(args.output/'smoke_adapter/classifier.npz')
                error=abs(saved.predict_failure_probability(vector,input_mode='raw')-probability)
                assert error<1e-5,'Saved head and training head disagree.'
                np.savez_compressed(args.output/'roundtrip_vector.npz',vector=vector)
                dump(args.output/'roundtrip_sample.json',positive)
                dump(args.output/'training_status.json',dict(state='smoke_passed',steps=step,lora_change_l2=delta,
                    adapter_sha256=identity,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                    saved_head_probability_error=error))
                handle.remove();return
            if not args.smoke_steps and (step%20==0 or start+2>=len(order)):
                result=validate();print('VALIDATION',step,{k:v for k,v in result.items() if k!='prefix_predictions'},flush=True)
                dump(args.output/f'validation_step_{step}.json',result)
                select(result,step)
                if stale>=2:break
        if stale>=2:break
    handle.remove()
    dump(args.output/'training_status.json',dict(state='complete',steps=step,best_step=best_step,best_validation_loss=best_loss))


if __name__=='__main__':
    main()
