def compute_token_spans(
    *,
    observation,
    prefix_tokens,
):
    spans = {"image": {}}

    cur = 0

    # image spans
    for image_name in observation.images:
        img_tokens = 256  # SigLIP patches
        spans["image"][image_name] = (cur, cur + img_tokens)
        cur += img_tokens

    images_end = cur

    if observation.tokenized_prompt is not None:
        tok_len = observation.tokenized_prompt.shape[1]

        if observation.task_token_len is not None:
            task_len = int(observation.task_token_len)

            spans["task"] = (
                images_end,
                images_end + task_len,
            )

            state_len = int(observation.state_token_len or 0)

            if state_len > 0:
                spans["state"] = (
                    images_end + task_len,
                    images_end + task_len + state_len,
                )

            spans["tokenized_prompt"] = (
                images_end,
                images_end + tok_len,
            )

        else:
            prompt_len = int(observation.tokenized_prompt_mask[0].sum())

            spans["task"] = (
                images_end,
                images_end + prompt_len,
            )

    spans["prefix"] = (
        0,
        int(prefix_tokens.shape[1]),
    )

    return spans