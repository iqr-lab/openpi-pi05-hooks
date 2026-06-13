def compute_token_spans(*, observation, prefix_tokens):
    spans = {"image": {}}
    cur = 0

    for image_name in observation.images:
        image_token_count = 256
        spans["image"][image_name] = {
            "start": cur,
            "end": cur + image_token_count,
        }
        cur += image_token_count

    if observation.task_token_len is not None:
        spans["prompt"] = {
            "start": cur,
            "end": cur + observation.task_token_len,
        }
        cur += observation.task_token_len

    if observation.state_token_len is not None:
        spans["state"] = {
            "start": cur,
            "end": cur + observation.state_token_len,
        }
        cur += observation.state_token_len

    spans["prefix"] = {
        "start": 0,
        "end": int(prefix_tokens.shape[1]),
    }

    return spans