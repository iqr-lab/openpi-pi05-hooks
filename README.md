# openpi-pi05-hooks

This repo is a fork of [openpi](https://github.com/Physical-Intelligence/openpi) for collecting internal model data during π0.5 inference.

The goal is to extract intermediate model information (embeddings, hidden states, attention, gradients, etc.) while preserving normal policy behavior.

---

# High-level flow

```text
serve_policy.py
  loads hooks.yaml

Policy.infer()
  calls model.sample_actions()

Pi0.sample_actions()
  runs normal inference

  exposes intermediate tensors
      ↓

runtime.collect_hook_data(...)
  computes enabled hook data

emit_all(...)
  formats hook records

PolicyRecorder
  saves:
      inputs
      outputs
      hook_records

LIBERO evaluation
  saves:
      episodes.json
```

The overall design separates:

```text
pi0.py
    exposes tensors

runtime.py
    computes derived quantities

hooks/
    formats outputs

PolicyRecorder
    saves results
```

This keeps model code relatively clean while allowing new hooks to be added easily.

---

# Folder structure

```text
src/pi05_hooks/

  hook_runner.py
  runtime.py

  computations/
    attention_utils.py
    token_spans.py
    action_chunks.py
    prefix_gradients.py
    raw_attention.py

  hooks/
    __init__.py
    observation_input.py
    token_spans.py
    prefix_embeddings.py
    prefix_final_hidden_state.py
    prefix_gradients.py
    action_chunks.py
    raw_attention_weights.py

configs/
  hooks.yaml
```

---

# Components

## hook_runner.py

Owns the global hook registry.

Provides:

```python
set_enabled_hooks(...)
set_hook_config(...)
get_hook_config()

register_hook(...)
is_hook_enabled(...)
emit_all(...)
```

Hooks register themselves using:

```python
register_hook("hook_name", hook_fn)
```

---

## runtime.py

Bridge between OpenPI internals and the hook framework.

`Pi0.sample_actions()` exposes intermediate tensors and calls:

```python
hook_data = collect_hook_data(...)
```

`runtime.py` decides which expensive computations should run:

```python
if is_hook_enabled("prefix_gradients"):
    data["prefix_gradients"] = compute_prefix_gradients(...)
```

This keeps most hook-specific logic outside of the model.

---

## computations/

Contains expensive computations.

Examples:

```python
compute_prefix_gradients(...)
compute_raw_attention_weights(...)
compute_action_chunks(...)
compute_token_spans(...)
```

These functions may perform:

* additional forward passes
* gradient computations
* attention extraction
* attribution calculations

---

## hooks/

Contains lightweight formatting functions.

Example output:

```python
{
    "hook_name": "prefix_gradients",
    "data": {...},
    "metadata": {...},
}
```

These hooks should not perform expensive computation.

---

# Configuration

Hooks are configured through a YAML file.

For a conceptual and tensor-level explanation of each π0.5 hook, see
[docs/pi05_hooks_technical.md](/Users/annsong/Desktop/openpi-pi05-hooks/docs/pi05_hooks_technical.md).

Example:

```yaml
record:
  enabled: true
  dir: /nfs/roberts/scratch/pi_tkf6/as4643/policy_records
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
    num_chunks: 8

  raw_attention_weights:
    layers: [1, 16]

  value_vectors:
    layers: [1, 16]
```

Server startup:

```bash
python scripts/serve_policy.py \
  --hook-config configs/hooks.yaml
```

---

# Output structure

Policy records:

```text
policy_records_pi05_libero_20260613_101500/

├── step_0.npy
├── step_1.npy
├── step_2.npy
└── ...
```

Evaluation metadata:

```text
data/libero/

├── episodes.json
└── videos/
```

Example episode metadata:

```json
{
  "global_episode_num": 0,
  "task_id": 0,
  "task": "put both the alphabet soup and the tomato sauce in the basket",
  "start_idx": 0,
  "end_idx": 55,
  "success": true,
  "num_policy_calls": 56,
  "num_env_steps": 214
}
```

This allows mapping:

```text
episode
    ↓
step indices
    ↓
hook records
```

---

# Hook lifecycle

A hook has three stages:

```text
Model
  ↓
runtime.py
  ↓
hook formatter
  ↓
saved record
```

---

# Current hooks

## observation_input

Stores the model input after preprocessing.

Includes:

* images
* image masks
* state
* prompt
* prompt mask
* token lengths

---

## token_spans

Stores token index ranges inside the prefix sequence.

Example:

```python
{
    "image": {
        "base_0_rgb": {"start": 0, "end": 256},
        "left_wrist_0_rgb": {"start": 256, "end": 512},
        "right_wrist_0_rgb": {"start": 512, "end": 768},
    },
    "prompt": {"start": 768, "end": 810},
    "state": {"start": 810, "end": 826},
    "prefix": {"start": 0, "end": 826},
}
```

This is used to map attention and attribution results back to:

* image tokens
* prompt tokens
* state tokens

---

## prefix_embeddings

Stores prefix embeddings before the transformer forward pass.

Shape:

```text
[batch, prefix_tokens, embedding_dim]
```

---

## prefix_final_hidden_state

Stores the final hidden state after the prefix forward pass.

Shape:

```text
[batch, prefix_tokens, hidden_dim]
```

---

## prefix_gradients

Stores:

```text
d(sum(first_step_flow_field))
-----------------------------
d(prefix_embeddings)
```

Shape:

```text
[batch, prefix_tokens, embedding_dim]
```

Used together with:

```text
token_spans
```

to attribute first-step flow-field predictions to:

* image tokens
* prompt tokens
* state tokens

---

## action_chunks

Stores one or more sampled action chunks.

Shape:

```text
[num_action_chunks, batch, action_horizon, action_dim]
```

If:

```yaml
num_chunks: 1
```

then only the normal action prediction is saved.

If:

```yaml
num_chunks: 8
```

then seven additional chunks are sampled using different flow-matching noise.

---

## raw_attention_weights

Stores raw transformer attention weights.

Shape:

```text
[batch, layers, heads, suffix_tokens, key_tokens]
```

Layer selection is controlled by:

```yaml
raw_attention_weights:
  layers: [1, 16]
```

or:

```yaml
raw_attention_weights:
  layers: all
```

---

## value_vectors

Stores the value vectors paired with the π0.5 prefix keys used by suffix attention.

They are read directly from the prefix KV cache, so this hook does not run an additional transformer pass.

Shape:

```text
[batch, layers, key_tokens, kv_heads, head_dim]
```

Layer selection is controlled independently from attention-weight recording:

```yaml
value_vectors:
  layers: [1, 16]
```

---

# Adding a new hook

There are three common cases.

---

## Case 1: data already exists

Example:

```python
prefix_mask
```

already exists in `hook_data`.

Create:

```text
hooks/prefix_mask.py
```

```python
from pi05_hooks.hook_runner import register_hook


def prefix_mask_hook(data):
    mask = data.get("prefix_mask")
    if mask is None:
        return None

    return {
        "hook_name": "prefix_mask",
        "data": {
            "mask": mask,
        },
    }


register_hook("prefix_mask", prefix_mask_hook)
```

Register:

```python
from pi05_hooks.hooks import prefix_mask  # noqa: F401
```

Enable:

```yaml
hooks:
  enabled:
    - prefix_mask
```

No runtime changes required.

---

## Case 2: data requires extra computation

Example:

```python
action_variance
```

Add:

```text
computations/action_variance.py
```

```python
import jax.numpy as jnp


def compute_action_variance(*, action_chunks):
    return jnp.var(action_chunks, axis=0)
```

Update runtime:

```python
if is_hook_enabled("action_variance"):
    data["action_variance"] = compute_action_variance(
        action_chunks=data["action_chunks"]
    )
```

Add formatter:

```text
hooks/action_variance.py
```

Register:

```python
register_hook("action_variance", action_variance_hook)
```

Enable:

```yaml
hooks:
  enabled:
    - action_chunks
    - action_variance
```

---

## Case 3: data is not available in hook_data

Sometimes the required tensor does not exist outside the model.

Examples:

```text
prefix_kv_cache
suffix_hidden_states
vision_features
vision_attention
```

In this case:

### Step 1

Expose the tensor inside `Pi0.sample_actions()`.

Example:

```python
hook_inputs = {
    "prefix_tokens": prefix_tokens,
    "prefix_mask": prefix_mask,
}
```

### Step 2

Pass it into:

```python
collect_hook_data(...)
```

### Step 3

Perform computation inside:

```text
runtime.py
```

### Step 4

Create a normal hook formatter.

Rule of thumb:

```text
pi0.py
    exposes tensors

runtime.py
    computes derived quantities

hooks/
    formats outputs
```

---

# External usage

The hook framework can be used outside OpenPI.

Example:

```python
from pi05_hooks.runtime import collect_hook_data
from pi05_hooks.hook_runner import emit_all
```

```python
hook_data = collect_hook_data(
    model=my_model,
    prefix_tokens=prefix_tokens,
    ...
)

records = emit_all(hook_data)
```

As long as the required tensors are provided, the hook framework does not depend on OpenPI.

---

# Future hooks

Potential future additions:

```text
prefix_kv_cache
suffix_hidden_states
vision_features
vision_attention

gradcam
integrated_gradients
activation_patching
attention_rollout
```

These can be added without modifying:

```text
PolicyRecorder
serve_policy.py
evaluation scripts
```

Only:

```text
pi0.py
runtime.py
hooks/
```

would need changes.
