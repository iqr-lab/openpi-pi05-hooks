from pi05_hooks.hook_runner import register_hook


def observation_input_hook(data):
    obs = data["observation"]

    return {
        "hook_name": "observation_input",
        "data": {
            "images": obs.images,
            "image_masks": obs.image_masks,
            "state": obs.state,
            "tokenized_prompt": obs.tokenized_prompt,
            "tokenized_prompt_mask": obs.tokenized_prompt_mask,
        },
        "metadata": {
            "image_names": list(obs.images.keys()),
            "state_shape": list(obs.state.shape),
            "tokenized_prompt_shape": (
                list(obs.tokenized_prompt.shape)
                if obs.tokenized_prompt is not None
                else None
            ),
        },
    }


register_hook("observation_input", observation_input_hook)