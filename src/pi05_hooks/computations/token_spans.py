def compute_token_spans(
    *,
    observation,
    prefix_tokens,
):
    spans = {}

    cur = 0

    for image_name in observation.images:
        img_tokens = 256  # siglip patches
        spans[image_name] = (cur, cur + img_tokens)
        cur += img_tokens

    if observation.tokenized_prompt is not None:
        prompt_len = int(observation.tokenized_prompt_mask[0].sum())
        spans["prompt"] = (cur, cur + prompt_len)

    return spans