"""Candidate questions for the overhead camera (logitech_2): hatch & probe and screws."""
import json, sys
sys.path.insert(0, '/home/steribot/projects/AD/anomaly_detection')
from botfails_benchmark.prompts import CAMERA
def yn(q, holds): return dict(question=f'{CAMERA} {q}\nAnswer with one word: yes or no.', options=['yes', 'no'], holds=holds, camera='logitech_2')
def ch(lead, a, b, c, holds): return dict(question=f'{CAMERA} {lead}\nA) {a}.\nB) {b}.\nC) {c}.\nAnswer with one letter: A, B or C.', options=['A', 'B', 'C'], holds=holds, camera='logitech_2')
v = {'industrial_robothon_hatchAndProbe': {
  'hatch_idle': ch('Look at the grey metal hatch on the black box and at the red robot gripper. Which describes the image?', 'The hatch is closed', 'The hatch is open and the red gripper is close to it', 'The hatch is open and the red gripper is far away from it', ['A', 'B']),
  'hatch_open': yn('Is the grey metal hatch on the black box open?', ['no']),
  'probe_socket': ch('Look at the two round sockets in the middle of the black box, one red and one black. Which describes the image?', 'The grey probe stands in the black socket', 'The grey probe stands in the red socket', 'The grey probe is not in any socket', ['A', 'C']),
  'probe_red': yn('Does the grey probe stand in the red socket?', ['no']),
 }, 'industrial_screws_sorting': {
  'plates_type': ch('Look at the two white plates at the top of the image. Which describes the screws on them?', 'The left plate holds only long screws or nothing, and the right plate holds only short screws or nothing', 'A long screw lies on the right plate, or a short screw lies on the left plate', 'No screw lies on the plates', ['A', 'C']),
  'long_right': yn('Is a long screw on the white plate at the top right?', ['no']),
  'short_left': yn('Is a short screw on the white plate at the top left?', ['no']),
  'on_table': yn('Is a screw lying on the wooden table, outside the plates and the yellow tray?', ['no']),
 }}
json.dump(v, open(sys.argv[1], 'w'), indent=1)
print({t: list(q) for t, q in v.items()})
