import json, sys
def yn(q, holds, crop): return dict(question=f'This image shows part of a table seen from above. {q}\nAnswer with one word: yes or no.', options=['yes', 'no'], holds=holds, crop=crop)
v = {'domotic_groceriesSorting': {
  'crop_sweet_in_left': yn('It shows a clear plastic box. Is there a sweet in the box, such as chocolate, a waffle, a biscuit or a donut?', ['no'], '#0,0.03,0.36,0.72'),
  'crop_produce_in_right': yn('It shows a clear plastic box. Is there a fruit or a vegetable in the box, such as a banana, grapes, an apple or a pepper?', ['no'], '#0.60,0.03,1,0.80'),
 }, 'domotic_vegetablesAndFruitsSorting': {
  'crop_veg_in_left': yn('It shows a clear plastic box. Is there a vegetable in the box, such as corn, broccoli, a pepper, a cucumber or lettuce?', ['no'], '#0,0.03,0.40,0.70'),
  'crop_fruit_in_right': yn('It shows a clear plastic box. Is there a fruit in the box, such as a banana, grapes, an apple, a lemon, an orange or a strawberry?', ['no'], '#0.66,0.03,1,0.85'),
 }}
json.dump(v, open(sys.argv[1], 'w'), indent=1); print({t: list(q) for t, q in v.items()})
