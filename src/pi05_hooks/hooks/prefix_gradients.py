from pi05_hooks.hook_runner import register_hook


def prefix_gradients_hook(data):
    gradients = data.get("prefix_gradients")

    if gradients is None:
        return None

    return {
        "hook_name": "prefix_gradients",
        "data": {
            "gradients": gradients,
        },
        "metadata": {
            "shape": list(gradients.shape),
            "meaning": "d(sum action velocity at t=1) / d(prefix_embeddings)",
        },
    }


register_hook("prefix_gradients", prefix_gradients_hook)