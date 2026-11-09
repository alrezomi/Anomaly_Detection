"""Task phases from the SO-100 gripper, the BotFails stand-in for our recorded stages.

Our robot reports its stages (1 approach, 2 grasp, 3 release). BotFails has no
stage messages, so the gripper defines them causally (only data up to t):

- a hold starts when the leader-arm command has been closed for DWELL_SEC and the
  follower fingers stopped above EMPTY_POS, i.e. an object blocks them;
- it ends when the command opens for DWELL_SEC, or when the fingers close fully
  while the command stays closed (the object slipped out).

Closing an empty gripper (operators park the arm with a closed gripper) is not a
hold. Thin objects (plates, cutlery, screws) can close the fingers almost fully;
those grasps are missed and the phase stays 'released', so the check asks the
placement question instead of the hold question. The constants come from the
normal training demonstrations only: at rest the closed fingers read <= 0.7
(95th percentile per task), a jug, spoon, iron, probe or cup stops them at 1-15.
"""
from __future__ import annotations

from bisect import bisect_right

import numpy as np

CLOSE_CMD, OPEN_CMD = 6., 8.  # Leader gripper command below/above which it counts as closed/open.
DWELL_SEC = .3                 # A command must last this long.
EMPTY_POS = .9                 # Settled closed fingers above this hold an object.
STILL_POS = .3                 # Fingers moved less than this over SETTLE_FRAMES: settled.
SETTLE_FRAMES = 6
MEDIAN_FRAMES = 9
FPS = 30.


def gripper_events(position, command, fps: float = FPS) -> list[tuple[float, str]]:
    """Causal ('hold' | 'release', time) events from 30 Hz gripper position and command."""
    position, command = np.asarray(position, float), np.asarray(command, float)
    dwell = int(round(DWELL_SEC * fps))
    events, holding, closed_run, open_run = [], False, 0, 0
    for i in range(len(command)):
        closed_run = closed_run + 1 if command[i] < CLOSE_CMD else 0
        open_run = open_run + 1 if command[i] > OPEN_CMD else 0
        recent = position[max(0, i - MEDIAN_FRAMES + 1):i + 1]
        if not holding:
            settled = closed_run >= dwell and i >= SETTLE_FRAMES and abs(position[i] - position[i - SETTLE_FRAMES]) < STILL_POS
            if settled and np.median(recent) > EMPTY_POS:
                holding = True
                events.append((i / fps, 'hold'))
        elif open_run >= dwell or (closed_run and len(recent) == MEDIAN_FRAMES and np.all(recent < EMPTY_POS - STILL_POS)):
            holding = False
            events.append((i / fps, 'release'))
    return events


class Phases:
    """Phase of the task at time t: index (changes at every event) and kind."""
    def __init__(self, events):
        self.events = list(events)
        self.times = [t for t, _ in self.events]

    def at(self, t: float) -> tuple[int, str]:
        index = bisect_right(self.times, t + 1e-9)
        if index == 0:
            return 0, 'start'          # Nothing grasped yet: like our stage 1, no sub-goal to check.
        return index, 'holding' if self.events[index - 1][1] == 'hold' else 'released'

    def summary(self):
        return [dict(time_sec=round(t, 3), event=kind) for t, kind in self.events]


def episode_phases(episode) -> Phases:
    robot = episode.robot()
    return Phases(gripper_events(robot['state'][:, 5], robot['action'][:, 5]))
