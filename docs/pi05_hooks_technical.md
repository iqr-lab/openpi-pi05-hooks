# π0.5 hook technical documentation

This document describes the hook set in this repository for the π0.5 model.

## π0.5 model context

π0.5 is a vision-language-action model built from two interacting transformer streams:

- a PaliGemma-style VLM backbone for images, language, and tokenized robot state;
- a smaller action expert for continuous low-level action generation.

For the low-level action path, the model builds a prefix from observations and language, caches its transformer keys/values, then repeatedly runs an action suffix through the action expert during flow matching. The suffix contains the current noisy/partially denoised action chunk. The action expert predicts a vector field, and the sampler integrates that vector field for `num_steps` denoising steps.

The paper describes π0.5 as using:

- image observations, language commands, and tokenized proprioceptive state as prefix information;
- an action expert that receives noisy continuous action tokens;
- adaptive RMSNorm to inject the flow-matching timestep into action-expert layers;
- attention masks that allow action-expert tokens to attend to the prefix and to one another, while preventing VLM tokens from attending back into the action expert;
- continuous action chunks produced by flow matching at inference time.

In this repository, the hook system captures the low-level `Pi0.sample_actions()` path. It does not currently hook the separate high-level autoregressive subtask generation path. So the hooks answer questions like “what did the low-level action expert attend to while producing this action chunk?” rather than “why did the high-level policy choose this subtask?”.

One implementation nuance: π0.5 supports tokenized proprioceptive state in the prefix when `discrete_state_input=True`. Some local configs disable that, in which case the state span is absent and the prefix contains images plus task/prompt tokens.

## Hook capture timing

The main capture point is inside [src/openpi/models/pi0.py](/Users/annsong/Desktop/openpi-pi05-hooks/src/openpi/models/pi0.py), after the normal action chunk has been sampled.

The sequence is:

1. preprocess the observation;
2. build prefix embeddings with `embed_prefix`;
3. run the prefix through the VLM/action transformer stack and cache prefix keys/values;
4. denoise the sampled action noise into a final action chunk;
5. call `collect_hook_data(...)`;
6. format enabled hook records outside JIT with `emit_all(...)`;
7. optionally save records through `PolicyRecorder`.

Expensive hooks are gated by `hooks.enabled`, so disabled hooks do not run their extra forward/gradient passes.

## Current hook audit

| Hook | Code location | Conceptual target | Status |
| --- | --- | --- | --- |
| `observation_input` | `src/pi05_hooks/hooks/observation_input.py` | Raw/preprocessed model inputs before embedding | Correct; now records masks and token-length metadata too. |
| `token_spans` | `src/pi05_hooks/computations/token_spans.py` | Mapping from prefix token indices to modalities | Correct for current π0.5 prefix layout. |
| `prefix_embeddings` | `src/pi05_hooks/hooks/prefix_embeddings.py` | Continuous prefix token embeddings before transformer layers | Correct. |
| `prefix_final_hidden_state` | `src/pi05_hooks/hooks/prefix_final_hidden_state.py` | Final prefix hidden states after prefix prefill | Correct. |
| `prefix_gradients` | `src/pi05_hooks/computations/prefix_gradients.py` | Local saliency of first-step action vector field w.r.t. prefix embeddings | Correct, but should be interpreted as first-flow-step saliency, not full final-action gradient. |
| `action_chunks` | `src/pi05_hooks/computations/action_chunks.py` | Final sampled action chunks and their initial noises | Correct. |
| `raw_attention_weights` | `src/pi05_hooks/computations/raw_attention.py` | First-step action-suffix attention onto prefix keys | Correct; layer indexing is JAX-safe and key slicing is π0.5 prefix-only. |
| `value_vectors` | `src/pi05_hooks/computations/value_vectors.py` | Prefix value vectors from the cached transformer KV state | Correct. |

## `observation_input`

### What it captures

This hook records the structured observation object passed into the model after policy-side preprocessing and batching:

- `images`: camera tensors, usually `[batch, height, width, channels]`, in the model image range;
- `image_masks`: per-camera validity masks;
- `state`: numeric robot state, `[batch, action_dim]`;
- `tokenized_prompt`: token IDs for language and, for π0.5 configs with discrete state input, tokenized proprioceptive state;
- `tokenized_prompt_mask`: validity mask for the tokenized prompt;
- `task_token_len` and `state_token_len`: static metadata used to split task-language tokens from state tokens;
- `prompt`: backwards-compatible alias for `tokenized_prompt`.


## `token_spans`

### What it captures

This hook returns index ranges into the prefix sequence:

```text
{
  "image": {
    camera_name: [start, end],
    ...
  },
  "task": [start, end],
  "state": [start, end],
  "tokenized_prompt": [start, end],
  "prefix": [0, prefix_len]
}
```

For the current image encoder, each image contributes 256 visual tokens. After image tokens, the tokenized prompt contributes language task tokens and, when available, discrete state tokens.

## `prefix_embeddings`

### What it captures

This hook records the prefix embeddings returned by `model.embed_prefix(observation)`.

Shape:

```text
[batch, prefix_tokens, embedding_dim]
```

For the default π0.5/PaliGemma 2B backbone in this repository, `embedding_dim` is 2048.


### Typical use

Use this hook when you want to:

- compare prefix representation norms across modalities;
- run offline clustering/probing over image vs text vs state embeddings;
- pair embeddings with `prefix_gradients` for gradient-times-activation style attribution.

## `prefix_final_hidden_state`

### What it captures

This hook records the final hidden state from the prefix-only transformer prefill.

Shape:

```text
[batch, prefix_tokens, embedding_dim]
```

## `prefix_gradients`

### What it captures

This hook recomputes a differentiable first-step action-expert pass and returns:

```text
d sum(v_t) / d prefix_embeddings
```

where `v_t` is the action expert’s predicted flow-matching vector field at the initial denoising point:

- action suffix input: the initial Gaussian noise passed to `sample_actions`;
- timestep: `τ = 1`;
- target scalar: the sum of the predicted vector field over batch, horizon, and action dimensions.

Shape:

```text
[batch, prefix_tokens, embedding_dim]
```

### Why it matters conceptually

π0.5 generates continuous actions through flow matching. The action expert does not directly emit final actions in a single transformer call; it emits a vector field that is integrated over denoising steps. This hook asks:

> If I infinitesimally changed each prefix embedding, how would the first predicted action-flow field change?

That makes it a local saliency signal for the low-level action expert’s initial flow direction.


## `action_chunks`

### What it captures

This hook records the final sampled action chunk and the initial noise that produced it.

Data fields:

```text
chunks: [num_chunks, batch, action_horizon, action_dim]
noises: [num_chunks, batch, action_horizon, action_dim]
```

With:

```yaml
action_chunks:
  num_chunks: 1
```

the hook saves only the normal action prediction from the policy call. If `num_chunks > 1`, the hook samples additional independent Gaussian noises and re-runs denoising for each one, producing multiple action chunks for the same prefix.

### Maming detail

`num_chunks: 1` does not mean one single action. It means one sampled rollout chunk. The number of actions inside that chunk is `model.action_horizon`.

For example:

```text
num_chunks = 1
action_horizon = 10
```

means:

```text
1 chunk containing 10 actions
```

## `raw_attention_weights`

### What it captures

This hook records transformer attention probabilities from action suffix queries to prefix keys during the first flow-matching denoising step.

Shape:

```text
[batch, layers, heads, suffix_tokens, key_tokens]
```

For π0.5:

- `suffix_tokens = action_horizon`, because the suffix is the continuous action-token sequence;
- `key_tokens = prefix_len`, because proprioceptive state is already tokenized into the prefix;
- `heads` is the number of query heads after grouped-query expansion;
- `layers` is either all layers or the layer subset configured in YAML.

### Why it matters conceptually

The π0.5 paper highlights the attention matrix as the mechanism that controls information flow between prefix observations/language/state and action tokens. This hook directly exposes the action expert’s attention distribution over the prefix.

It answers questions like:

- Which camera tokens do action tokens attend to?
- Does the first action token attend differently from later action tokens?
- Do deeper action-expert layers shift attention from language to visual/state tokens?
- Which prefix regions are attended by particular action dimensions or timesteps?

### Capture details

This hook performs an additional suffix forward pass with `record_attn=True`. It uses:

- the same prefix KV cache produced by the normal prefix prefill;
- the initial action noise;
- timestep `τ = 1`;
- the π0.5 suffix mask, where action tokens can attend to prefix tokens and to each other.

The hook records prefix-key attention only. It intentionally slices away suffix-to-suffix keys so the saved tensor focuses on how action tokens read observation/task/state context.

## `value_vectors`

### What it captures

This hook records the cached value vectors for prefix keys.

Shape:

```text
[batch, layers, key_tokens, kv_heads, head_dim]
```

For the default Gemma configs in this repository:

```text
kv_heads = 1
head_dim = 256
```

This hook is the direct counterpart to `raw_attention_weights`:

- `raw_attention_weights`: where the action suffix looked;
- `value_vectors`: what content was available at those looked-at locations.

### Capture details

This hook does not run another transformer pass. It reads the value half of the prefix KV cache returned by the prefix prefill.

The computation selects layers independently from `raw_attention_weights`:

```yaml
value_vectors:
  layers: null  # all layers
```

or:

```yaml
value_vectors:
  layers: [1, 16]
```

## Recommended YAML profiles

### Full interpretability capture


```yaml
record:
  enabled: true
  add_timestamp: true

hooks:
  enabled:
    - observation_input
    - token_spans
    - prefix_embeddings
    - prefix_final_hidden_state
    - prefix_gradients
    - action_chunks
    - raw_attention_weights
    - value_vectors

  action_chunks:
    num_chunks: 1

  raw_attention_weights:
    layers: null

  value_vectors:
    layers: null
```

### Smaller attention-focused capture

Use this when recording all layers is too heavy:

```yaml
record:
  enabled: true
  add_timestamp: true

hooks:
  enabled:
    - observation_input
    - token_spans
    - action_chunks
    - raw_attention_weights
    - value_vectors

  action_chunks:
    num_chunks: 1

  raw_attention_weights:
    layers: [1, 16]

  value_vectors:
    layers: [1, 16]
```
