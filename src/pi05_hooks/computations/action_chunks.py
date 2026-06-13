import jax
import jax.numpy as jnp

from pi05_hooks.hook_runner import get_hook_config


def compute_action_chunks(*, rng, model, actions, run_denoising):
    cfg = get_hook_config().get("action_chunks", {})
    num_chunks = int(cfg.get("num_chunks", 1))

    if num_chunks <= 1:
        return actions[None, ...]

    batch_size = actions.shape[0]

    extra_noises = jax.random.normal(
        rng,
        (
            num_chunks - 1,
            batch_size,
            model.action_horizon,
            model.action_dim,
        ),
    )

    chunks = [actions]
    for i in range(num_chunks - 1):
        chunks.append(run_denoising(extra_noises[i]))

    return jnp.stack(chunks, axis=0)