import json, sys
sys.path.insert(0, '/home/steribot/projects/AD/anomaly_detection')
from botfails_benchmark.prompts import CAMERA
def yn(name, q, holds, phases=('holding', 'released')):
    return dict(name=name, phases=list(phases), question=f'{CAMERA} {q}\nAnswer with one word: yes or no.', options=['yes', 'no'], holds=holds)
def ch(name, lead, a, b, c, holds, phases=('holding', 'released')):
    return dict(name=name, phases=list(phases), question=f'{CAMERA} {lead}\nA) {a}.\nB) {b}.\nC) {c}.\nAnswer with one letter: A, B or C.', options=['A', 'B', 'C'], holds=holds)
FRUITS, VEG = 'a banana, grapes, an apple, a lemon, an orange or a strawberry', 'corn, broccoli, a pepper, a cucumber or lettuce'
q = {
 'domotic_vegetablesAndFruitsSorting': [
   yn('fruit_named_in_yellow', f'Is {FRUITS} in the clear box with the yellow-green mark?', ['no']),
   yn('veg_named_in_pink', f'Is {VEG} in the clear box with the pink mark?', ['no']),
   ch('pink_box', 'Look at the clear box with the pink mark. Which describes it?', f'It is empty or holds only fruits: {FRUITS}', f'It holds a vegetable: {VEG}', 'It is not visible', ['A', 'C']),
   ch('yellow_box', 'Look at the clear box with the yellow-green mark. Which describes it?', f'It is empty or holds only vegetables: {VEG}', f'It holds a fruit: {FRUITS}', 'It is not visible', ['A', 'C'])],
 'industrial_screws_sorting': [
   ch('front_plate', 'Look at the white plate in front. Which describes it?', 'It holds only long bolts, or nothing', 'It holds a short screw', 'It is not visible', ['A', 'C'])],
 'industrial_soldering': [
   ch('iron', 'Where is the soldering iron?', 'In its spiral holder', 'Lying somewhere else', 'Held by the red robot gripper', ['A', 'C'])],
}
json.dump(q, open(sys.argv[1], 'w'), indent=1)
