"""Sub-goal questions derived from the nominal reference runs (rynnbrain.vlm_check.questions = "reference").

The fixed questions (prompts.subgoal_questions) are written by hand for one setup: they name the object, the
destination and the gripper ("held between the two white fingers", "lies on a blue surface"). Here the base VLM
writes the questions itself, per task, from the nominal runs the detectors are calibrated on, so a new task or
dataset needs no hand-written wording:

1. Describe. At gripper-camera frames of a few reference runs where a sub-goal visibly holds (stage 2: the
   object is held; stage 3: it has just been released at the destination), the VLM names the object, says where
   it is after the release, describes the moment in one sentence, and writes a yes/no question that checks the
   step. It does this twice: told the task in one sentence (rynnbrain.vlm_check.task_instruction, else
   rynnbrain.task_description), written once per task group ("Pick up the USB plug and insert it into its
   matching hole on the board."), and not told it. Told the task it names the part correctly (without it
   RynnBrain said "nut", "hinge" or "pipe" for plugs), but its own visual words ("the square object") made the
   better check on some tasks, so both kinds of wording compete in step 3.
2. Ask back. The answers become yes/no questions: "Is <object> held between the gripper fingers?",
   "Is <object> <where it is>?", the questions it wrote, and "Does this sentence describe the image correctly?
   <sentence>". The task sentence is not part of these questions: a prompt that states what should be true
   gets answered by repeating it.
3. Check on every reference run. Each candidate is asked the way the live check asks it (mean over the frames
   CHECK_OFFSETS_SEC before the check time) where the sub-goal holds (positives: during stage 2 for the grasp,
   after the release for the placement) and on single frames where it does not hold yet (negatives: stage 1
   before the grasp; the carrying part of stage 2 for the placement). Its threshold is the one that best
   separates the two (Youden's J: share of positives kept plus share of negatives stopped, minus 1; the lower
   threshold on ties), never above 0.5, the VLM's own yes/no point.
4. Choose. The candidate with the highest J (then AUROC) is the stage's question, with that threshold. A
   question that hardly separates (AUROC below MIN_AUROC) is still used, flagged, and its threshold set to keep
   GUARD_KEEP of the normal checks: what the VLM cannot verify must not stop normal runs. When item and destination
   are configured, the fixed question is scored on the same frames for comparison only (on REASSEMBLE Ethernet
   it answered "not held" on 78% of the normal frames where the plug was held); reference mode never falls
   back to it.

   A first version (v3) set the threshold below the lowest positive, so that no normal reference check would
   stop. That rule is degenerate whenever one normal frame gets a low answer: on REASSEMBLE several stages
   could then stop nothing at all (BNC placement: 0% of the "not done yet" frames), and on the benchmark it
   stopped 5 of 37 true alarms. With J it stopped 15 of 37, still ignoring all 20 false alarms.

A description of one run's details (a position, a hallucinated colour) has to hold on every reference run, so
it does not survive step 3. Only nominal runs are used: failures are never seen, so step 3 can only reject a
question that does not tell "done" from "not done yet"; whether it catches failures is for the benchmark.
The result (questions, every candidate's scores, the descriptions) is saved as JSON next to the DINO memory
(or rynnbrain.vlm_check.questions_file) and reused while the reference runs, task sentence, camera and model
stay the same; every question's reference scores are also kept in <file>_scores.json, so deriving again only
asks what is new.

    python -m rynnbrain_vlm.reference_questions --config pipeline_config.json [--rebuild | --reselect]

--reselect applies the current choice rule to the candidate scores saved in the file, without the VLM.
"""
import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERSION = 'reference_questions_v6'
CAMERA = ('This image comes from a camera mounted on a robot gripper. It looks down past the gripper fingers at what '
          'is below the gripper.')
YES_NO = '\nAnswer with one word: yes or no.'
# No example answers in the prompts: given one ("on the table"), the VLM repeated it for every run.
NAME_PROMPT = 'What object is held between the gripper fingers? Answer with a short noun phrase.'
PLACE_PROMPTS = ('The gripper has just let go of the object. Where is that object now? Answer with a short phrase that '
                 'starts with a preposition.',
                 'What is the object below the gripper resting on or inserted into? Answer with a short phrase that '
                 'starts with a preposition.')
SENTENCE_PROMPTS = {2: 'Describe in one short sentence what the gripper is holding and how.',
                    3: 'The gripper has just let go of the object. In one short sentence, where is that object now?'}
WRITE_PROMPTS = {2: 'The robot has just grasped the object. Write one short yes/no question about this image that '
                    'checks whether the grasp worked. Write only the question.',
                 3: 'The robot has just let go of the object. Write one short yes/no question about this image that '
                    'checks whether the object is where the task wants it. Write only the question.'}
YES_NO_STARTS = ('is', 'are', 'does', 'do', 'has', 'have', 'can', 'was', 'were')
DESCRIBED_RUNS = 6  # Reference runs whose frames the VLM describes (spread over the list).
CHECKS = 3  # Positive check times and negative frames per run and sub-goal.
MAX_THRESHOLD = .5  # A derived question never stops when the VLM leans to "yes".
MIN_AUROC = .75  # Below this the chosen question hardly tells "done" from "not done yet"...
GUARD_KEEP = .95  # ...so its threshold keeps this share of the normal reference checks.
PREPOSITIONS = {'in', 'on', 'inside', 'into', 'onto', 'at', 'under', 'below', 'beneath', 'next', 'near', 'beside',
                'between', 'above', 'over', 'within', 'inserted', 'plugged', 'resting', 'lying', 'placed'}


def held_question(noun):
    return f'{CAMERA} Is {noun} held between the gripper fingers?{YES_NO}'


def placed_question(noun, place):
    return f'{CAMERA} Is {noun} {place}?{YES_NO}'


def sentence_question(sentence):
    return f'{CAMERA} Does this sentence describe the image correctly?\n"{sentence}"{YES_NO}'


def written_question(question):
    return f'{CAMERA} {question}{YES_NO}'


def describe_prompt(prompt, instruction):
    """A describing prompt, told the robot's task when there is one."""
    return f"{CAMERA} The robot's task: {instruction} {prompt}" if instruction else f'{CAMERA} {prompt}'


def ungrounded(text):
    """RynnBrain's pointing format removed: tags such as <object> <area> ... </area> and coordinates go; a name it
    marked, "<usb plug>" or "[USB plug]", keeps its words."""
    text = re.sub(r'</?(?:object|area|frame|point|points|box|bbox|traj|trajectory|grasp pose|ref)>', ' ', str(text))
    text = re.sub(r'\(\s*\d+\s*,\s*\d+\s*\)', ' ', text)
    return re.sub(r'<([^<>]{1,40})>', r'\1', text).replace('[', '').replace(']', '')


def yes_no_question(text):
    """A question the VLM wrote, if it is a short plain yes/no question."""
    text = ungrounded(text).strip().strip('"\'')
    text = re.sub(r'\s+', ' ', text.splitlines()[0] if text else '').strip()
    text = re.sub(r'[\s,;]+$', '', text.split('?')[0]) + '?' if '?' in text else ''
    words = text.lower().split()
    if not words or words[0] not in YES_NO_STARTS or len(words) > 20 or re.search(r"[^A-Za-z '?,-]", text):
        return None
    return text[0].upper() + text[1:]


def clean(text):
    """First sentence of a VLM answer, without its pointing format."""
    text = ungrounded(text).strip().strip('"\'')
    text = text.splitlines()[0] if text else ''
    text = re.split(r'(?<=[.!?])\s', text)[0]
    return re.sub(r'\s+', ' ', re.sub(r'\s+([,.;:])', r'\1', text)).strip(' .,;:')


def noun_phrase(text):
    text = clean(text).lower()
    text = re.sub(r'^(it is|this is|there is|the (robot|gripper) is holding|the gripper holds)\s+', '', text)
    words = text.split()
    # Plain words only: RynnBrain sometimes answers in its grounding format ("<grasp pose> (383, 430), ...").
    if not words or len(words) > 6 or re.search(r"[^a-z '-]", text):
        return None
    return text if words[0] in ('a', 'an', 'the') else 'the ' + text


def place_phrase(text):
    text = clean(text).lower()
    text = re.sub(r'^(it is|it\'s|the [a-z ]+? (is|lies|sits))\s+(now\s+)?', '', text)
    words = text.split()
    if not words or len(words) > 8 or words[0] not in PREPOSITIONS or re.search(r"[^a-z '-]", text):
        return None
    # "Is <object> <place>?" must read as a question: "into the box" -> "in the box", "onto" -> "on".
    return re.sub(r'^(into|onto)\b', lambda m: m.group(1)[:2], text)


def settings(config):
    return config['rynnbrain'].get('vlm_check') or {}


def reference_bags(config):
    return list(settings(config).get('reference_bags') or config['nominal_bags'])


def task_instruction(config):
    """One sentence on what the robot does, the same for a whole task group; '' when none is configured."""
    text = ' '.join(str(settings(config).get('task_instruction') or config['rynnbrain'].get('task_description')
                        or '').split())
    return text if not text or text.endswith(('.', '!', '?')) else text + '.'


def questions_path(config):
    if settings(config).get('questions_file'):
        return Path(settings(config)['questions_file'])
    return Path(config['nominal_cache_dir']) / 'vlm_questions' / f"{config['vision_memory_name']}.json"


def signature(config):
    from time_series_gmr_scripts import run_time_series_test as gmr_settings
    from .cascade import CHECK_OFFSETS_SEC
    check = settings(config)
    return dict(version=VERSION, model=config['rynnbrain']['model']['model_id'], camera=check.get('camera_topic'),
                stage_topic=config.get('stage_topic', '/recording_stage'), references=reference_bags(config),
                instruction=task_instruction(config), item=check.get('item'), destination=check.get('destination'),
                check_offsets_sec=list(CHECK_OFFSETS_SEC), post_stage3_sec=gmr_settings.POST_STAGE3_SEC)


def question_set(saved, config):
    from .cascade import SUBGOAL_THRESHOLD
    from .prompts import QuestionSet
    check = settings(config)
    derived = {int(goal): stage['chosen'] for goal, stage in saved['stages'].items() if stage.get('use')}
    return QuestionSet(check.get('item'), check.get('destination'), derived, source='reference',
                       default_threshold=SUBGOAL_THRESHOLD)


def load_questions(config):
    """The saved question set, or None when it is missing or was derived for other runs, camera or model."""
    path = questions_path(config)
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('signature') != json.loads(json.dumps(signature(config))):
        return None
    return question_set(saved, config)


def load_or_derive(config, model, progress=print):
    questions = load_questions(config)
    if questions is not None:
        progress(f'VLM check questions derived from the reference runs: {questions_path(config)}')
        return questions
    saved = derive(config, model, progress)
    return question_set(saved, config)


def reference_frames(bag, stage_topic, topic, offsets, post_sec, max_size):
    """Gripper-camera images of one nominal run: per sub-goal, positive checks (one image per offset) and
    negative frames. The latest frame at or before each time is used, as in the live check."""
    from rosbag_io import decode_image_message
    from .cascade_benchmark import camera_frames, stage_boundaries
    boundaries = stage_boundaries(bag, stage_topic)
    if not {1, 2, 3} <= set(boundaries):
        raise ValueError('the run lacks stage 1, 2 or 3')
    b1, b2, b3 = boundaries[1], boundaries[2], boundaries[3]
    if b2 - b1 < 2.5 or b3 - b2 < 2.:
        raise ValueError('stage 1 or 2 is too short for reference frames')
    times = {2: dict(pos=np.linspace(b2 + 1., b3 - .5, CHECKS), neg=np.linspace(b1 + .5, b2 - 1.5, CHECKS)),
             3: dict(pos=np.linspace(b3 + 1., b3 + post_sec - .5, CHECKS), neg=np.linspace(b2 + 1., (b2 + b3) / 2, CHECKS))}
    last = max(max(t['pos'].max(), t['neg'].max()) for t in times.values())
    frames = camera_frames(bag, stage_topic, topic, last)
    if not frames:
        raise ValueError('no gripper-camera frames')
    stamps, decoded = [t for t, _ in frames], {}

    def image(t):
        position = max(0, int(np.searchsorted(stamps, t + 1e-9, side='right')) - 1)
        if position not in decoded:
            source, payload = frames[position][1]
            picture = decode_image_message(*payload) if source == 'message' else payload
            picture = picture.convert('RGB')
            picture.thumbnail((max_size, max_size))
            decoded[position] = picture
        return decoded[position]

    return {goal: dict(pos=[[image(t - o) for o in offsets] for t in sets['pos']], neg=[image(t) for t in sets['neg']])
            for goal, sets in times.items()}


def auroc(positives, negatives):
    p, n = np.asarray(positives)[:, None], np.asarray(negatives)[None, :]
    return float(np.mean((p > n) + .5 * (p == n)))


def derive(config, model, progress=print):
    """Describe, ask back, check on every reference run, choose; save and return the JSON record."""
    from time_series_gmr_scripts import run_time_series_test as gmr_settings
    from .cascade import CHECK_OFFSETS_SEC
    from .prompts import subgoal_questions
    started = time.monotonic()
    check = settings(config)
    topic, stage_topic = check['camera_topic'], config.get('stage_topic', '/recording_stage')
    references, skipped = [], {}
    for bag in reference_bags(config):
        try:
            references.append((bag, reference_frames(bag, stage_topic, topic, CHECK_OFFSETS_SEC,
                                                     gmr_settings.POST_STAGE3_SEC, model.max_image_size)))
        except Exception as error:
            skipped[bag] = str(error)
    if len(references) < 2:
        raise ValueError(f'Need at least 2 usable reference runs to derive questions; skipped: {skipped}')
    progress(f'Deriving the VLM check questions from {len(references)} reference runs'
             + (f' ({len(skipped)} skipped)' if skipped else ''))

    def ask(question, options, picture):
        return model.answer_probability(dict(role='user', images=[('Gripper camera image:', picture)], text=question),
                                        options)

    def say(prompt, picture):
        return model.generate_nominal(dict(role='user', images=[('Gripper camera image:', picture)], text=prompt),
                                      {'max_new_tokens': 40, 'do_sample': False})

    # 1. Describe a few runs, spread over the reference list: told the task (when there is a task sentence) and
    # not told it. Told the task, the VLM names the part correctly ("Ethernet plug"), but its own visual words
    # ("the square object") made better checks on some tasks, so both kinds compete in step 3.
    instruction = task_instruction(config)
    picks = sorted({int(round(i)) for i in np.linspace(0, len(references) - 1, min(DESCRIBED_RUNS, len(references)))})
    top = lambda counter, k: [text for text, _ in counter.most_common(k)]
    candidates, descriptions, first_noun = {2: [], 3: []}, {}, None
    for mode, sentence in [('task', instruction)] * bool(instruction) + [('plain', '')]:
        prompt = lambda text: describe_prompt(text, sentence)
        nouns, places, raw = Counter(), Counter(), []
        sentences, written = {2: Counter(), 3: Counter()}, {2: Counter(), 3: Counter()}
        for index in picks:
            bag, images = references[index]
            held, released = images[2]['pos'][CHECKS // 2][0], images[3]['pos'][0][0]
            answers = dict(bag=bag, name=say(prompt(NAME_PROMPT), held),
                           places=[say(prompt(text), released) for text in PLACE_PROMPTS],
                           sentence_2=say(prompt(SENTENCE_PROMPTS[2]), held),
                           sentence_3=say(prompt(SENTENCE_PROMPTS[3]), released),
                           question_2=say(prompt(WRITE_PROMPTS[2]), held),
                           question_3=say(prompt(WRITE_PROMPTS[3]), released))
            raw.append(answers)
            for counter, value in ((nouns, noun_phrase(answers['name'])),
                                   *((places, place_phrase(p)) for p in answers['places']),
                                   (sentences[2], clean(answers['sentence_2'])),
                                   (sentences[3], clean(answers['sentence_3'])),
                                   (written[2], yes_no_question(answers['question_2'])),
                                   (written[3], yes_no_question(answers['question_3']))):
                if value:
                    counter[value] += 1
            progress(f"Described {Path(bag).name} ({mode}): {answers['name']!r} | "
                     + ' | '.join(map(repr, answers['places'])) + f" | {answers['question_2']!r} | {answers['question_3']!r}")
        first_noun = first_noun or (top(nouns, 1) or [None])[0]
        descriptions[mode] = dict(raw=raw, nouns=dict(nouns), places=dict(places),
                                  sentences={str(g): dict(c) for g, c in sentences.items()},
                                  written={str(g): dict(c) for g, c in written.items()})
        # 2. Ask back: candidate questions per sub-goal.
        candidates[2] += ([(f'name ({mode})', held_question(n)) for n in top(nouns, 3) + ['the object']]
                          + [(f'written ({mode})', written_question(q)) for q in top(written[2], 2)]
                          + [(f'sentence ({mode})', sentence_question(s)) for s in top(sentences[2], 2)])
        candidates[3] += ([(f'name + place ({mode})', placed_question(n, p))
                           for n in top(nouns, 1) + ['the object'] for p in top(places, 3)]
                          + [(f'written ({mode})', written_question(q)) for q in top(written[3], 2)]
                          + [(f'sentence ({mode})', sentence_question(s)) for s in top(sentences[3], 2)])
    fixed = {}
    if check.get('item') and check.get('destination'):
        fixed = {goal: subgoal_questions(goal, check['item'], check['destination'])[0] for goal in (2, 3)}

    # 3. Check every candidate (and, for comparison, the fixed question) on every reference run. Scores are
    # cached next to the question file, so deriving again after a change only asks what is new.
    cache = ScoreCache(config, model.max_image_size)

    def scores(question, options, goal):
        def compute():
            positives = [float(np.mean([ask(question, options, picture) for picture in pictures]))
                         for _, images in references for pictures in images[goal]['pos']]
            negatives = [ask(question, options, picture) for _, images in references for picture in images[goal]['neg']]
            return [round(p, 4) for p in positives], [round(n, 4) for n in negatives]
        return cache.get(question, options, goal, compute)

    stages = {}
    for goal in (2, 3):
        rows, seen = [], set()
        for kind, question in candidates[goal]:
            if question in seen:
                continue
            seen.add(question)
            positives, negatives = scores(question, ('yes', 'no'), goal)
            rows.append(dict(kind=kind, question=question, options=['yes', 'no'], positives=positives,
                             negatives=negatives))
            progress(f'Stage {goal} candidate {len(rows)} scored: {question[len(CAMERA) + 1:].splitlines()[0]}')
        reference_fixed = None
        if goal in fixed:  # For comparison only.
            question, options = fixed[goal]
            positives, negatives = scores(question, options, goal)
            reference_fixed = dict(question=question, options=list(options), positives=positives, negatives=negatives)
        stages[str(goal)] = dict(candidates=rows, fixed=reference_fixed)
    cache.save()
    saved = dict(signature=json.loads(json.dumps(signature(config))), created_utc=datetime.now(timezone.utc).isoformat(),
                 references=[bag for bag, _ in references], skipped=skipped, instruction=instruction,
                 object=first_noun or 'the object', descriptions=descriptions, stages=stages,
                 elapsed_sec=round(time.monotonic() - started, 1))
    return choose_and_save(saved, config, progress)


class ScoreCache:
    """Reference scores of every question ever asked for these reference runs, camera and model
    (<question file>_scores.json), keyed by question, options and sub-goal."""
    def __init__(self, config, max_image_size):
        from time_series_gmr_scripts import run_time_series_test as gmr_settings
        from .cascade import CHECK_OFFSETS_SEC
        path = questions_path(config)
        self.path = path.with_name(path.stem + '_scores.json')
        self.identity = dict(model=config['rynnbrain']['model']['model_id'], camera=settings(config).get('camera_topic'),
                             stage_topic=config.get('stage_topic', '/recording_stage'),
                             references=reference_bags(config), check_offsets_sec=list(CHECK_OFFSETS_SEC),
                             post_stage3_sec=gmr_settings.POST_STAGE3_SEC, checks=CHECKS, max_image_size=max_image_size,
                             frames='reference_frames_v1')
        self.identity = json.loads(json.dumps(self.identity))
        saved = json.loads(self.path.read_text(encoding='utf-8')) if self.path.is_file() else {}
        self.scores = saved.get('scores', {}) if saved.get('identity') == self.identity else {}

    @staticmethod
    def key(question, options, goal):
        return json.dumps([question, list(options), int(goal)])

    def get(self, question, options, goal, compute):
        key = self.key(question, options, goal)
        if key not in self.scores:
            positives, negatives = compute()
            self.scores[key] = dict(positives=positives, negatives=negatives)
            self.save()
        return self.scores[key]['positives'], self.scores[key]['negatives']

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(identity=self.identity, scores=self.scores)), encoding='utf-8')


def separation(positives, negatives, cap=MAX_THRESHOLD):
    """Youden threshold on the reference checks (stop below it) and how well it separates."""
    positives, negatives = np.asarray(positives, float), np.asarray(negatives, float)
    cuts = np.r_[np.unique(np.r_[positives, negatives]), max(positives.max(), negatives.max()) + 1e-6]
    j = lambda t: np.mean(positives >= t) + np.mean(negatives < t) - 1
    threshold = float(min(cap, max(cuts, key=lambda t: (j(t), -t))))
    return dict(threshold=threshold, j=float(j(threshold)), kept=float(np.mean(positives >= threshold)),
                specificity=float(np.mean(negatives < threshold)), auroc=auroc(positives, negatives))


def choose_and_save(saved, config, progress=print):
    """4. Choose per stage the candidate with the best J (then AUROC); score the fixed question for comparison."""
    from .cascade import SUBGOAL_THRESHOLD
    for goal, stage in saved['stages'].items():
        rows = [dict(row, **separation(row['positives'], row['negatives'])) for row in stage['candidates']]
        best = max(rows, key=lambda r: (r['j'], r['auroc']))
        separates = best['auroc'] >= MIN_AUROC
        if not separates:
            # It cannot tell done from not done on the normal runs, so it must not stop normal runs either: J alone
            # once chose a cut that stopped 77% of the normal releases (BNC). Keep GUARD_KEEP of the normal checks.
            p, n = np.asarray(best['positives']), np.asarray(best['negatives'])
            threshold = float(min(MAX_THRESHOLD, np.quantile(p, 1 - GUARD_KEEP)))
            best = dict(best, threshold=threshold, kept=float(np.mean(p >= threshold)),
                        specificity=float(np.mean(n < threshold)), j=float(np.mean(p >= threshold) + np.mean(n < threshold) - 1),
                        guarded=True)
        fixed = stage.get('fixed')
        if fixed:
            p, n = np.asarray(fixed['positives']), np.asarray(fixed['negatives'])
            fixed = dict({k: v for k, v in fixed.items() if k in ('question', 'options', 'positives', 'negatives')},
                         threshold=SUBGOAL_THRESHOLD, kept=float(np.mean(p >= SUBGOAL_THRESHOLD)),
                         specificity=float(np.mean(n < SUBGOAL_THRESHOLD)), auroc=auroc(p, n),
                         youden=separation(p, n))
        saved['stages'][goal] = dict(chosen=dict(best, separates=separates), use=True, candidates=rows, fixed=fixed)
        progress(f"Stage {goal}: {best['question'][len(CAMERA) + 1:].splitlines()[0]} (stop below "
                 f"{best['threshold']:.2f}: keeps {100 * best['kept']:.0f}% of normal checks, stops "
                 f"{100 * best['specificity']:.0f}% of not-done frames, AUROC {best['auroc']:.2f}"
                 + ('' if separates else f', hardly separates: set to keep {100 * GUARD_KEEP:.0f}% of normal checks') + ')'
                 + (f"; fixed question at 0.5: keeps {100 * fixed['kept']:.0f}%, stops {100 * fixed['specificity']:.0f}%,"
                    f" AUROC {fixed['auroc']:.2f}" if fixed else ''))
    saved['signature'] = json.loads(json.dumps(signature(config)))
    saved['method'] = dict(described_runs=DESCRIBED_RUNS, checks=CHECKS, threshold='youden', max_threshold=MAX_THRESHOLD,
                           min_auroc=MIN_AUROC, guard_keep=GUARD_KEEP)
    path = questions_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=1), encoding='utf-8')
    progress(f'Saved the derived questions: {path}')
    return saved


def reselect(config, progress=print):
    """Apply the current choice rule to the candidate scores saved by an earlier version (no VLM needed).

    Only the version may differ: the reference runs, camera and model must be the ones the scores came from."""
    saved = json.loads(questions_path(config).read_text(encoding='utf-8'))
    current = json.loads(json.dumps(signature(config)))
    if {k: v for k, v in saved['signature'].items() if k != 'version'} != {k: v for k, v in current.items() if k != 'version'}:
        raise ValueError('The saved candidates were scored for other reference runs, camera or model; use --rebuild.')
    saved['reselected_from'] = saved['signature']['version']
    saved['stages'] = {goal: dict(candidates=[{k: v for k, v in row.items()
                                               if k in ('kind', 'question', 'options', 'positives', 'negatives')}
                                              for row in stage['candidates']], fixed=stage.get('fixed'))
                       for goal, stage in saved['stages'].items()}
    return choose_and_save(saved, config, progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path('pipeline_config.json'))
    parser.add_argument('--rebuild', action='store_true', help='Derive again even if a matching file exists.')
    parser.add_argument('--reselect', action='store_true',
                        help='Choose again from the saved candidate scores with the current rule (no VLM).')
    arguments = parser.parse_args()
    from pipeline_settings import load_pipeline_config
    config = load_pipeline_config(arguments.config)
    if arguments.reselect:
        reselect(config, lambda message: print(message, flush=True))
        return
    if not arguments.rebuild and load_questions(config) is not None:
        print(f'Up to date: {questions_path(config)} (use --rebuild to derive again)')
        return
    from .model import RynnBrainModel
    vlm = config['rynnbrain']
    model = RynnBrainModel({**vlm['model'], 'lora_adapter_path': None, **settings(config).get('model', {})})
    derive(config, model, lambda message: print(message, flush=True))


if __name__ == '__main__':
    main()
