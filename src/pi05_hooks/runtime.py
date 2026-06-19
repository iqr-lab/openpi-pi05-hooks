from pi05_hooks.computations.action_chunks import compute_action_chunks
from pi05_hooks.computations.prefix_gradients import compute_prefix_gradients
from pi05_hooks.computations.raw_attention import compute_raw_attention_weights
from pi05_hooks.computations.token_spans import compute_token_spans
from pi05_hooks.computations.value_vectors import compute_value_vectors
from pi05_hooks.hook_runner import is_hook_enabled


def collect_hook_data(
    *,
    model,
    rng,
    observation,
    prefix_tokens,
    prefix_mask,
    prefix_ar_mask,
    prefix_final_hidden_state,
    kv_cache,
    noise,
    actions,
    run_denoising,
):
    data = {
        "observation": observation,
        "prefix_tokens": prefix_tokens,
        "prefix_mask": prefix_mask,
        "prefix_ar_mask": prefix_ar_mask,
        "prefix_final_hidden_state": prefix_final_hidden_state,
        "kv_cache_after_prefix": kv_cache,
        "actions": actions,
    }

    if is_hook_enabled("token_spans"):
        data["token_spans"] = compute_token_spans(
            observation=observation,
            prefix_tokens=prefix_tokens,
        )

    if is_hook_enabled("action_chunks"):
        data["action_chunks"] = compute_action_chunks(
            rng=rng,
            model=model,
            actions=actions,
            noise=noise,
            run_denoising=run_denoising,
        )

    if is_hook_enabled("prefix_gradients"):
        data["prefix_gradients"] = compute_prefix_gradients(
            model=model,
            observation=observation,
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_ar_mask=prefix_ar_mask,
            noise=noise,
        )

    if is_hook_enabled("raw_attention_weights"):
        data["raw_attention_weights"] = compute_raw_attention_weights(
            model=model,
            observation=observation,
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_ar_mask=prefix_ar_mask,
            kv_cache=kv_cache,
            noise=noise,
        )

    if is_hook_enabled("value_vectors"):
        data["value_vectors"] = compute_value_vectors(
            prefix_tokens=prefix_tokens,
            kv_cache=kv_cache,
        )

    return data
