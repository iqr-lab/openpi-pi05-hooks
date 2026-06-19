from pi05_hooks.hook_runner import register_hook


@register_hook("value_vectors")
def emit(data):
    value_data = data["value_vectors"]
    return {
        "hook_name": "value_vectors",
        "data": {
            "vectors": value_data["vectors"],
            "layers": value_data["layers"],
        },
    }
