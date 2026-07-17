from pi05_hooks.hook_runner import register_hook


@register_hook("observation_input")
def emit(data):
    obs = data["observation"]

    return {
        "hook_name": "observation_input",
        "data": {
            "images": obs.images,
            "image_masks": obs.image_masks,
            "state": obs.state,
            "tokenized_prompt": getattr(
                obs,
                "tokenized_prompt",
                None,
            ),
            "tokenized_prompt_mask": getattr(
                obs,
                "tokenized_prompt_mask",
                None,
            ),
            "task_token_len": getattr(
                obs,
                "task_token_len",
                None,
            ),
            "state_token_len": getattr(
                obs,
                "state_token_len",
                None,
            ),
            # Backwards-compatible alias used by earlier records.
            "prompt": getattr(
                obs,
                "tokenized_prompt",
                None,
            ),
        },
    }
