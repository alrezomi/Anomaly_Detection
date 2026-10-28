"""Observation scheduling for browser VLM replay; independent of model training."""

import math


def replay_settings(config):
    vlm = config.get("rynnbrain", {})
    options = vlm.get("online_replay", {})
    if not isinstance(options, dict):
        raise ValueError("rynnbrain.online_replay must be an object.")
    mode = options.get("interval_mode", "constant")
    if mode not in ("constant", "adaptive"):
        raise ValueError("online_replay.interval_mode must be constant or adaptive.")
    limit = options.get("max_observations", vlm.get("num_frames", 8))
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("online_replay.max_observations must be a positive integer or null.")
    stop = options.get("stop_on_alarm", False)
    if not isinstance(stop, bool):
        raise ValueError("online_replay.stop_on_alarm must be true or false.")
    return dict(interval_mode=mode, max_observations=limit, stop_on_alarm=stop)


def observation_interval(settings, score, constant_interval):
    if settings["interval_mode"] == "constant":
        return float(constant_interval)
    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Adaptive replay requires an available logistic failure score in [0, 1].")
    if score < .3:
        return 10.
    if score < .5:
        return 7.
    if score < .6:
        return 5.
    return 3.
