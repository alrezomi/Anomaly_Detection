#!/bin/bash
echo 'Archived hybrid experiment. Use final.sh for the strict DINO -> VLM cascade.' >&2
exit 1
S=/tmp/claude-1000/-home-steribot-projects/f0e2de92-2e5a-401c-8559-5de8ed21cbd2/scratchpad
I=/evaluation/improvement; E=/evaluation
for spec in "cv_vlm|" "cv_vlm_rel|--release-delay 1.5"; do
  IFS='|' read -r name extra <<< "$spec"
  DOCKER_NAME=botfails-cascade $S/botfails_docker.sh python -u run_botfails_benchmark.py cascade --dataset $E/dataset/BotFails --frames $E/frames --dino-run $I/dino/best --tasks-config $I/cv_tasks_nodecision.json --vlm-thresholds $I/final_thresholds.json --recheck-sec 2 $extra --output $I/final/$name --split test --vlm-cache $I/vlm_cache.jsonl > /home/steribot/projects/AD/evaluation/FIDeL/improvement/final/$name.log 2>&1
  echo "done $name"
done
echo FINISHED
