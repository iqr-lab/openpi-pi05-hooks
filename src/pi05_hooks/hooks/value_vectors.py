from pi05_hooks.hook_runner import register_hook


@register_hook("value_vectors")
def emit(data):
    value_data = data["value_vectors"]
    return {
        "hook_name": "value_vectors",
        "data": {
            "vectors": value_data["vectors"],
            "layers": value_data["layers"],
            "key_end": value_data["key_end"],
            "num_kv_heads": value_data["num_kv_heads"],
            "head_dim": value_data["head_dim"],
        },
    }
