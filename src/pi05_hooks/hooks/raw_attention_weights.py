from pi05_hooks.hook_runner import register_hook


def raw_attention_weights_hook(data):
    raw_attention = data.get("raw_attention_weights")

    if raw_attention is None:
        return None

    weights = raw_attention["weights"]

    return {
        "hook_name": "raw_attention_weights",
        "data": {
            "weights": weights,
            "layers": raw_attention["layers"],
        }
    }


register_hook("raw_attention_weights", raw_attention_weights_hook)