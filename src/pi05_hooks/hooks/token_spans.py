from pi05_hooks.hook_runner import register_hook


def token_spans_hook(data):
    spans = data.get("token_spans")

    if spans is None:
        return None

    return {
        "hook_name": "token_spans",
        "data": spans,
        "metadata": {
            "meaning": (
                "Token index ranges for image, prompt, and state "
                "within prefix_tokens."
            )
        },
    }


register_hook("token_spans", token_spans_hook)