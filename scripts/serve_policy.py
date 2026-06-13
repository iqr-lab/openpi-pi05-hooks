import dataclasses
import enum
import logging
import os
import socket
from datetime import datetime

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
print("DEBUG: imported openpi modules", flush=True)

from pi05_hooks.hook_runner import set_enabled_hooks, set_hook_config
import pi05_hooks.hooks  # noqa: F401
print("DEBUG: imported pi05_hooks.hooks", flush=True)


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint("pi05_aloha", "gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.ALOHA_SIM: Checkpoint("pi0_aloha_sim", "gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint("pi05_droid", "gs://openpi-assets/checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _parse_hooks_from_env() -> list[str]:
    hooks_str = os.environ.get("PI05_HOOKS", "")
    return [h.strip() for h in hooks_str.split(",") if h.strip()]


def _parse_attn_layers_from_env() -> list[int] | None:
    layers_str = os.environ.get("PI05_ATTN_LAYERS", "all").strip().lower()
    if layers_str in ("", "all", "none"):
        return None
    return [int(x.strip()) for x in layers_str.split(",") if x.strip()]


def main(args: Args) -> None:
    print("=" * 80, flush=True)
    print("DEBUG: entered main()", flush=True)
    print("=" * 80, flush=True)
    print(f"DEBUG: args = {args}", flush=True)

    hooks = _parse_hooks_from_env()
    attn_layers = _parse_attn_layers_from_env()
    num_action_chunks = int(os.environ.get("PI05_NUM_ACTION_CHUNKS", "1"))

    print("DEBUG: configuring hooks", flush=True)
    print(f"DEBUG: hooks = {hooks}", flush=True)
    print(f"DEBUG: attention layers = {attn_layers}", flush=True)
    print(f"DEBUG: num_action_chunks = {num_action_chunks}", flush=True)

    set_enabled_hooks(hooks)
    set_hook_config(
        {
            "action_chunks": {
                "num_chunks": num_action_chunks,
            },
            "raw_attention_weights": {
                "layers": attn_layers,
            },
        }
    )

    print("DEBUG: creating policy", flush=True)
    policy = create_policy(args)
    print("DEBUG: policy created successfully", flush=True)
    print(f"DEBUG: policy type = {type(policy)}", flush=True)

    policy_metadata = policy.metadata
    print("DEBUG: retrieved metadata", flush=True)

    if args.record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        match args.policy:
            case Checkpoint():
                policy_tag = args.policy.config
            case Default():
                policy_tag = args.env.value

        record_dir = (
            f"/nfs/roberts/scratch/pi_tkf6/as4643/"
            f"policy_records_{policy_tag}_{timestamp}"
        )

        print(f"DEBUG: record_dir = {record_dir}", flush=True)
        policy = _policy.PolicyRecorder(policy, record_dir)
        print("DEBUG: PolicyRecorder created", flush=True)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"DEBUG: hostname = {hostname}", flush=True)
    print(f"DEBUG: local_ip = {local_ip}", flush=True)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )

    print("DEBUG: websocket server created", flush=True)
    print(f"DEBUG: listening on port {args.port}", flush=True)
    print("DEBUG: entering serve_forever()", flush=True)

    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    print("DEBUG: before tyro.cli", flush=True)
    parsed_args = tyro.cli(Args)
    print(f"DEBUG: parsed_args = {parsed_args}", flush=True)
    main(parsed_args)