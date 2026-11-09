# BotFails experiment scripts

The exact scripts behind the numbers in `evaluation/FIDeL/improvement/README.md`, kept as they were run. They have the
evaluation machine's paths near the top of each file, so adapt those before reusing them. The benchmark itself is
`run_botfails_benchmark.py`.

| Files | Purpose |
|---|---|
| `dev_split.json` | Development (episode index % 5 in {0, 1}) and held-out test episodes |
| `bf_vlm_server.py` | Keeps the VLM loaded and answers queued question jobs |
| `bf_nominal_job.py`, `bf_analyze.py`, `bf_questions.py`, `q_v2_gen.py`, `q_v3_gen.py`, `make_generic.py` | First round: questions tested on normal training frames |
| `dev_job.py`, `dev_analyze.py`, `per_episode.py` | Questions tested on development episodes |
| `gen1.py`, `candidate_task_descriptions.json`, `overhead_q.py`, `v3_gen.py` | Candidate questions: richer descriptions, overhead camera, box crops |
| `cal_job.py`, `cal_thresholds.py` | Per-question thresholds from normal training frames |
| `ref_job.py`, `positional.py`, `dino_compare.py` | Reference-image comparison, position-aware patch memory, camera comparison |
| `replay.sh`, `replay_table.py` | Cascade replay and report on development episodes |
| `final.sh`, `cv_runs.sh`, `cv_select.py`, `compare.py` | Final run, candidate rules, cross-validated rule choice, comparison table |
