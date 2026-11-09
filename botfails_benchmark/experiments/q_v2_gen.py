import json, sys
sys.path.insert(0, '/home/steribot/projects/AD/anomaly_detection')
from botfails_benchmark.prompts import CAMERA
def yn(name, q, holds, phases=('holding', 'released')):
    return dict(name=name, phases=list(phases), question=f'{CAMERA} {q}\nAnswer with one word: yes or no.', options=['yes', 'no'], holds=holds)
def ch(name, lead, a, b, c, holds, phases=('holding', 'released')):
    return dict(name=name, phases=list(phases), question=f'{CAMERA} {lead}\nA) {a}.\nB) {b}.\nC) {c}.\nAnswer with one letter: A, B or C.', options=['A', 'B', 'C'], holds=holds)
held = {'domotic_dishTidyUp': 'a cup, a plate or a whisk', 'domotic_groceriesSorting': 'a grocery item', 'domotic_makingCoffee': 'the spoon',
        'domotic_pouringCoffee': 'the jug', 'domotic_setTheTable': 'a plate, a glass, a fork or a knife', 'domotic_vegetablesAndFruitsSorting': 'a fruit or a vegetable',
        'industrial_robothon_hatchAndProbe': 'the red probe', 'industrial_screws_sorting': 'a screw', 'industrial_soldering': 'the soldering iron'}
q = {}
for task, obj in held.items():
    q[task] = [yn('hold_robot', f'Is the red robot holding {obj}?', ['yes'], ['holding']),
               ch('hold_choice', 'Look at the gripper of the red robot. Which describes it?', f'It holds {obj}', 'It holds nothing', 'The gripper is hidden', ['A', 'C'], ['holding'])]
q['domotic_dishTidyUp'] += [yn('plates_upright', 'Look at the metal dish rack. Does a plate lie flat in it instead of standing upright?', ['no']),
                            yn('dish_outside', 'Does a dish lie on its side next to the dish rack?', ['no'])]
q['domotic_vegetablesAndFruitsSorting'] += [yn('fruit_in_yellow', 'Is there a fruit in the clear box with the yellow-green mark?', ['no']),
                                            yn('veg_in_pink', 'Is there a vegetable in the clear box with the pink mark?', ['no'])]
q['domotic_groceriesSorting'] += [yn('sweet_in_pink', 'Is there a sweet, such as chocolate, a biscuit or a donut, in the clear box with the pink sign?', ['no']),
                                  yn('produce_in_yellow', 'Is there a fruit or a vegetable in the clear box with the yellow-green sign?', ['no'])]
q['industrial_screws_sorting'] += [yn('screw_on_table', 'Does a screw lie on the table outside the plates?', ['no'])]
json.dump(q, open(sys.argv[1], 'w'), indent=1)
print({t: [c['name'] for c in v] for t, v in q.items()})
