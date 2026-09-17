import jax.numpy as jnp

from pi05_hooks.hook_runner import get_hook_config

_REDUCTIONS = ("none", "mean_H", "mean_k", "mean_H_k")


def _cfg():
    return get_hook_config().get("suffix_final_hidden_state", {})


def suffix_hidden_buffer_steps():
    """Static size of the per-step buffer; must be >= the num_steps used at inference."""
    return int(_cfg().get("max_steps", 10))


def compute_suffix_final_hidden_state(*, hidden_states, num_steps):
    """Select and pool the action expert's final hidden states across denoising steps.

    hidden_states: [max_steps, batch, action_horizon, hidden_dim], one row per denoising
    step (step 0 is the pure-noise input, the last step produces the final action).
    """
    cfg = _cfg()
    reduce = cfg.get("reduce", "mean_H_k")
    if reduce not in _REDUCTIONS:
        raise ValueError(f"suffix_final_hidden_state.reduce must be one of {_REDUCTIONS}, got {reduce!r}")

    max_steps = hidden_states.shape[0]
    num_steps = jnp.minimum(jnp.asarray(num_steps, dtype=jnp.int32), max_steps)
    selected_steps = cfg.get("steps", "all")

    if selected_steps is None or selected_steps == "all":
        step_indices = jnp.arange(max_steps)
        valid = step_indices < num_steps
    else:
        # Negative indices count back from the last step actually run.
        requested = jnp.array(selected_steps, dtype=jnp.int32)
        step_indices = jnp.where(requested < 0, requested + num_steps, requested)
        valid = (step_indices >= 0) & (step_indices < num_steps)
        step_indices = jnp.clip(step_indices, 0, max_steps - 1)

    # [steps, batch, H, d] -> [batch, steps, H, d]
    hidden = hidden_states[step_indices].transpose(1, 0, 2, 3)
    step_weights = valid.astype(jnp.float32)[None, :, None, None]

    if reduce == "mean_H":
        hidden = hidden.mean(axis=2)
    elif reduce in ("mean_k", "mean_H_k"):
        pooled = (hidden.astype(jnp.float32) * step_weights).sum(axis=1)
        pooled = pooled / jnp.maximum(valid.sum(), 1)
        hidden = pooled.astype(hidden.dtype)
        if reduce == "mean_H_k":
            hidden = hidden.mean(axis=1)

    return {
        "hidden_states": hidden,
        "steps": step_indices,
        "valid_steps": valid,
        "num_steps": num_steps,
    }
