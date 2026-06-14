import jax
import jax.numpy as jnp

from pi05_hooks.hook_runner import get_hook_config


def compute_action_chunks(*, rng, model, actions, noise, run_denoising):
    cfg = get_hook_config().get("action_chunks", {})
    num_chunks = int(cfg.get("num_chunks", 1))

    chunks = [actions]
    noises = [noise]

    if num_chunks > 1:
        extra_rng = jax.random.fold_in(rng, 12345)

        extra_noises = jax.random.normal(
            extra_rng,
            (
                num_chunks - 1,
                actions.shape[0],
                model.action_horizon,
                model.action_dim,
            ),
        )

        for i in range(num_chunks - 1):
            n = extra_noises[i]
            noises.append(n)
            chunks.append(run_denoising(n))

    return {
        "chunks": jnp.stack(chunks, axis=0),
        "noises": jnp.stack(noises, axis=0),
    }