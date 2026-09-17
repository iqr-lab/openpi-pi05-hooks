from pi05_hooks.hook_runner import get_hook_config, register_hook


@register_hook("suffix_final_hidden_state")
def emit(data):
    out = dict(data["suffix_final_hidden_state"])
    out["reduce"] = get_hook_config().get("suffix_final_hidden_state", {}).get("reduce", "mean_H_k")
    return {
        "hook_name": "suffix_final_hidden_state",
        "data": out,
    }
