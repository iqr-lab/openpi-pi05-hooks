from pi05_hooks.hook_runner import register_hook


def final_action_chunk_hook(data):
    actions = data["actions"]

    return {
        "hook_name": "final_action_chunk",
        "data": {
            "actions": actions,
        },
        "metadata": {
            "shape": list(actions.shape),
            "meaning": "[batch, action_horizon, action_dim]",
        },
    }


register_hook("final_action_chunk", final_action_chunk_hook)