import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

from pi05_hooks.hook_runner import emit_all, is_hook_enabled, get_hook_config

logger = logging.getLogger("openpi")

# None = record all layers.
_ATTN_LAYERS = None


def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.action_in_proj = nnx.Linear(
            config.action_dim,
            action_expert_config.width,
            rngs=rngs,
        )

        if config.pi05:
            self.time_mlp_in = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            self.time_mlp_out = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
        else:
            self.state_proj = nnx.Linear(
                config.action_dim,
                action_expert_config.width,
                rngs=rngs,
            )
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )
            self.action_time_mlp_out = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                rngs=rngs,
            )

        self.action_out_proj = nnx.Linear(
            action_expert_config.width,
            config.action_dim,
            rngs=rngs,
        )

        self.deterministic = True

        # Needed for the raw-attention recording module.
        self._paligemma_config = paligemma_config
        self._action_expert_config = action_expert_config
        self._embed_dtype = config.dtype
        self._adarms = config.pi05

    def get_hook_records(self):
        return getattr(self, "_last_hook_records", [])

    def _get_llm_vars(self) -> dict:
        from flax.nnx.bridge import variables as bv  # noqa: PLC0415

        nnx_attrs = {
            name: getattr(self.PaliGemma.llm, name)
            for name in self.PaliGemma.llm.linen_attributes
        }
        return bv.nnx_attrs_to_linen_vars(nnx_attrs)

    def _make_record_module(self) -> _gemma.Module:
        return _gemma.Module(
            configs=[self._paligemma_config, self._action_expert_config],
            embed_dtype=self._embed_dtype,
            adarms=self._adarms,
            record_attn=True,
        )

    @at.typecheck
    def embed_prefix(
        self,
        obs: _model.Observation,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)

        time_emb = posemb_sincos(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )

        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)

            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(
                time_emb,
                "b emb -> b s emb",
                s=self.action_horizon,
            )
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)

            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))

        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            x_t,
            time,
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        _, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _compute_raw_attention_weights(
        self,
        observation,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        kv_cache,
        noise,
    ):
        batch_size = prefix_tokens.shape[0]
        prefix_len = prefix_tokens.shape[1]

        suf_tok, suf_mask, suf_ar_mask, adarms = self.embed_suffix(
            observation,
            noise,
            jnp.ones(batch_size),
        )

        suf_len = suf_tok.shape[1]
        pfx_for_suf = einops.repeat(prefix_mask, "b p -> b s p", s=suf_len)
        suf_attn = make_attn_mask(suf_mask, suf_ar_mask)
        full_attn = jnp.concatenate([pfx_for_suf, suf_attn], axis=-1)

        suf_pos = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suf_mask, axis=-1)
            - 1
        )

        record_module = self._make_record_module()
        variables = self._get_llm_vars()

        _outputs, _kv_new, attn_probs = record_module.apply(
            variables,
            [None, suf_tok],
            suf_pos,
            full_attn,
            [None, adarms],
            kv_cache=kv_cache,
        )

        has_suffix_state_token = not self.pi05
        key_end = prefix_len + (1 if has_suffix_state_token else 0)

        if _ATTN_LAYERS is None:
            layer_indices = jnp.arange(attn_probs.shape[0])
            attn_weights = attn_probs[:, :, :, :, :, :key_end]
        else:
            layer_indices = jnp.array(_ATTN_LAYERS)
            attn_weights = attn_probs[_ATTN_LAYERS, :, :, :, :, :key_end]

        num_layers = attn_weights.shape[0]
        k_heads = attn_weights.shape[2]
        g_groups = attn_weights.shape[3]

        attn_weights = attn_weights.reshape(
            num_layers,
            batch_size,
            k_heads * g_groups,
            suf_len,
            key_end,
        )

        attn_weights = attn_weights.transpose(1, 0, 2, 3, 4)

        return {
            "weights": attn_weights,
            "layers": layer_indices,
            "key_end": key_end,
            "suffix_len": suf_len,
            "num_heads": k_heads * g_groups,
            "num_layers": num_layers,
        }

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        def run_denoising(start_noise):
            def step(carry):
                x_t, time = carry

                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation,
                    x_t,
                    jnp.broadcast_to(time, batch_size),
                )

                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(
                    prefix_mask,
                    "b p -> b s p",
                    s=suffix_tokens.shape[1],
                )
                full_attn_mask = jnp.concatenate(
                    [prefix_attn_mask, suffix_attn_mask],
                    axis=-1,
                )

                assert full_attn_mask.shape == (
                    batch_size,
                    suffix_tokens.shape[1],
                    prefix_tokens.shape[1] + suffix_tokens.shape[1],
                )

                positions = (
                    jnp.sum(prefix_mask, axis=-1)[:, None]
                    + jnp.cumsum(suffix_mask, axis=-1)
                    - 1
                )

                prefix_out_step, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )

                assert prefix_out_step is None

                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

                return x_t + dt * v_t, time + dt

            def cond(carry):
                x_t, time = carry
                return time >= -dt / 2

            x_0, _ = jax.lax.while_loop(cond, step, (start_noise, 1.0))
            return x_0

        if noise is None:
            noise = jax.random.normal(
                rng,
                (batch_size, self.action_horizon, self.action_dim),
            )

        x_0 = run_denoising(noise)

        fiper_chunks = None
        if is_hook_enabled("fiper_action_chunks"):
            hook_config = get_hook_config()
            num_samples = int(hook_config.get("ace_num_samples", 8))

            sample_noises = jax.random.normal(
                rng,
                (num_samples, batch_size, self.action_horizon, self.action_dim),
            )

            chunks = []
            for i in range(num_samples):
                chunks.append(run_denoising(sample_noises[i]))

            fiper_chunks = jnp.stack(chunks, axis=0)

        raw_attention_weights = None
        if is_hook_enabled("raw_attention_weights"):
            raw_attention_weights = self._compute_raw_attention_weights(
                observation=observation,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                prefix_ar_mask=prefix_ar_mask,
                kv_cache=kv_cache,
                noise=noise,
            )

        hook_data = {
            "observation": observation,
            "prefix_tokens": prefix_tokens,
            "prefix_mask": prefix_mask,
            "prefix_ar_mask": prefix_ar_mask,
            "prefix_final_hidden_state": prefix_out,
            "kv_cache_after_prefix": kv_cache,
            "actions": x_0,
            "fiper_action_chunks": fiper_chunks,
            "raw_attention_weights": raw_attention_weights,
        }

        self._last_hook_records = emit_all(hook_data)

        return x_0