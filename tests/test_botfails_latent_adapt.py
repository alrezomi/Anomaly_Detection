import unittest
from types import SimpleNamespace

import numpy as np

from botfails_benchmark.latent_adapt import causal_context_frames, check_joint_head_context, metrics, outcome, score
from botfails_benchmark.latent_train import epoch_samples, paired_order


class AdaptationTests(unittest.TestCase):
    def test_every_training_update_is_class_balanced_and_task_matched(self):
        samples=[]
        for task in ('one','two'):
            samples.append(dict(task=task,bag=task+'/failure',target=1,bag_failure=True))
            samples.append(dict(task=task,bag=task+'/failure',target=0,bag_failure=True))
            samples.extend(dict(task=task,bag=task+'/nominal'+str(i),target=0,bag_failure=False) for i in range(8))
        order=paired_order(samples,42)
        self.assertEqual(len(order),16)
        for start in range(0,len(order),2):
            pair=order[start:start+2]
            self.assertEqual({s['target'] for s in pair},{0,1})
            self.assertEqual(len({s['task'] for s in pair}),1)
        self.assertEqual(sum(s['task']=='one' and s['target']==1 for s in order),4)

    def test_joint_head_rejects_an_adapter_or_reference_from_another_run(self):
        reference={'reference':'normal_train/task/episode_000000','response':'nominal'}
        metadata=dict(representation_id='rep',vector_dimension=4096,input_mode='raw',comparison_signature=dict(
            model_id='model',lora_adapter_sha256='adapter',context_sampling='causal_uniform',num_frames=8,reference_contexts={'task':reference}))
        check_joint_head_context(metadata,'task',reference,'adapter','model','causal_uniform','rep')
        with self.assertRaises(ValueError):
            check_joint_head_context(metadata,'task',reference,'other_adapter','model','causal_uniform','rep')
        with self.assertRaises(ValueError):
            check_joint_head_context(metadata,'task',{**reference,'response':'changed'},'adapter','model','causal_uniform','rep')

    def test_training_pair_targets_change_only_after_the_annotated_onset(self):
        ep=SimpleNamespace(key='test/task/episode_000000',task='task',split='test',name='episode_000000')
        frames=[(i*15,i*.5,str(i)) for i in range(21)]
        truth=np.r_[np.zeros(60),np.ones(241)]
        samples=epoch_samples(ep,frames,truth,1,0)
        self.assertEqual([s['target'] for s in samples],[0,1])
        self.assertLess(samples[0]['timestamp_sec'],2.)
        self.assertGreaterEqual(samples[1]['timestamp_sec'],2.)

    def test_training_does_not_invent_a_pre_failure_negative_for_initial_failure(self):
        ep=SimpleNamespace(key='test/task/episode_000000',task='task',split='test',name='episode_000000')
        frames=[(i*15,i*.5,str(i)) for i in range(21)]
        self.assertTrue(all(s['target']==1 for s in epoch_samples(ep,frames,np.ones(301),1,0)))

    def test_late_failure_without_an_available_camera_image_is_not_a_positive_prefix(self):
        ep=SimpleNamespace(key='test/task/episode_000000',task='task',split='test',name='episode_000000')
        frames=[(i*15,i*.5,str(i)) for i in range(21)]
        self.assertTrue(all(s['target']==0 for s in epoch_samples(ep,frames,np.r_[np.zeros(301),np.ones(2)],1,0)))

    def test_context_selection_cannot_use_the_future_recording_length(self):
        frames = [(i, i*.5, str(i)) for i in range(21)]
        before = causal_context_frames(frames[:13], 6.)
        after = causal_context_frames(frames, 6.)
        self.assertEqual(before, after)
        self.assertEqual(len(after), 8)
        self.assertTrue(all(1. <= f[1] <= 6. for f in after))
        self.assertEqual(after[-1][1], 6.)

    def test_first_context_uses_only_the_first_available_camera_image(self):
        frames = [(i, i*.5, str(i)) for i in range(10)]
        self.assertEqual(causal_context_frames(frames, 1.), [(2, 1., '2')])

    def test_first_early_alarm_cannot_be_rescued_by_later_failure_detection(self):
        record = {'observations': [{'timestamp_sec': 1., 'score': .8},
                                   {'timestamp_sec': 4., 'score': .9}]}
        row = outcome(record, np.r_[np.zeros(60), np.ones(90)], .75)
        self.assertFalse(row['correct'])
        result = metrics([row])
        self.assertEqual((result['early'], result['detected'], result['missed']), (1, 0, 0))

    def test_all_continue_fails_recall_gate_even_with_zero_false_alarms(self):
        record = {'observations': [{'timestamp_sec': 1., 'score': .1}]}
        result = metrics([outcome(record, np.ones(60), .75), outcome(record, None, .75)])
        self.assertEqual(result['recall'], 0.)
        self.assertEqual(result['specificity'], 1.)
        self.assertEqual(result['balanced_recording_accuracy'], .5)

    def test_knn_uses_different_nominal_bags_instead_of_duplicate_prefixes(self):
        head = dict(memory=np.array([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]]),
                    bag_ids=np.array([0, 0, 1, 2]), distance_threshold=2.)
        expected = .75 * (0 + np.sqrt(2) + 2) / 3 / 2
        self.assertAlmostEqual(score(head, np.array([2., 0.])), expected)

    def test_logistic_scoring_normalizes_and_centers_before_dot_product(self):
        head = dict(weights=np.array([2., -1.]), feature_mean=np.array([.1, .2]), intercept=.3)
        expected = 1 / (1 + np.exp(-((np.array([.6, .8]) - head['feature_mean']) @ head['weights'] + .3)))
        self.assertAlmostEqual(score(head, np.array([3., 4.])), expected)


if __name__ == '__main__':
    unittest.main()
