#!/bin/bash
# usage: replay.sh <name> <tasks.json inside container> [extra cascade args...]   (development episodes only)
S=/tmp/claude-1000/-home-steribot-projects/f0e2de92-2e5a-401c-8559-5de8ed21cbd2/scratchpad
name=$1; tasks=$2; shift 2
E=/evaluation; I=$E/improvement
scale=1.1; prev=""; for a in "$@"; do [ "$prev" = "--threshold-scale" ] && scale=$a; prev=$a; done
DOCKER_NAME=botfails-cascade $S/botfails_docker.sh sh -c "python -u run_botfails_benchmark.py cascade --dataset $E/dataset/BotFails --frames $E/frames --dino-run $I/dino/best --tasks-config $tasks --output $I/runs/$name --split test --split-file $I/dev_split.json --subset dev --vlm-cache $I/vlm_cache.jsonl $* > $I/runs/$name.log 2>&1; python -u run_botfails_benchmark.py report --dataset $E/dataset/BotFails --dino-run $I/dino/best --cascade-run $I/runs/$name --output $I/runs/$name/report --threshold-scale $scale --split-file $I/dev_split.json --subset dev --no-timelines 2>&1 | tail -1"
$S/venv/bin/python -I $S/exp2/replay_table.py /home/steribot/projects/AD/evaluation/FIDeL/improvement/runs/$name "$name"
