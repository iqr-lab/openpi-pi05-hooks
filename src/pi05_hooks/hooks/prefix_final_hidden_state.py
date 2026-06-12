from pi05_hooks.hook_runner import register_hook


def prefix_final_hidden_state_hook(data):
    hidden_state = data["prefix_final_hidden_state"]

    return {
        "hook_name": "prefix_final_hidden_state",
        "data": {
            "hidden_state": hidden_state,
        },
        "metadata": {
            "shape": list(hidden_state.shape),
            "meaning": "[batch, prefix_tokens, hidden_dim]",
        },
    }


register_hook("prefix_final_hidden_state", prefix_final_hidden_state_hook)