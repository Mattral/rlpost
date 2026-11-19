"""Command-line interface (CLI) for RLHF training pipelines.

Provides:
- train-sft: SFT on tiny_lm when ``--tiny`` is set. Does not download a model.
- train-reward: Reward model on tiny_lm when ``--tiny`` is set. Does not download a model.
- run-ppo: PPO on tiny_lm when ``--tiny`` is set. Does not download a model.
- run-dpo: DPO on tiny_lm when ``--tiny`` is set. Does not download a model.
- Global --toy flag for small-model testing on T4 (<20 minutes)
- Config validation before execution
"""

import logging
from typing import Optional

import typer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="rlpost",
    help=(
        "SFT, reward-model, PPO, and DPO commands. "
        "--tiny trains tiny_lm on CPU and does not download a model."
    ),
)


def setup_logging(verbose: bool = False) -> None:
    """Setup console logging.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def train_sft(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Directory for run.json. Required with --tiny.",
    ),
    tiny: bool = typer.Option(
        False,
        "--tiny",
        help="Train tiny_lm on CPU. Does not download a model.",
    ),
    steps: int = typer.Option(
        1,
        "--steps",
        help="SFT updates to take in this call. At least 1.",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        help="Rows per update.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="TrainingConfig.seed. Default 42.",
    ),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint file written at the end of this call.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Load --checkpoint before the first step.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging",
    ),
) -> None:
    """Train SFT on tiny_lm.

    ``--tiny`` builds one tiny Llama and trains it on a fixed list of
    prompt and completion strings. The loss ignores prompt tokens. It
    does not download a model. Pass ``--checkpoint`` to write a
    checkpoint at the end, and ``--resume`` to load that file first.

    Example:
        python -m rlpost.cli train-sft --tiny --steps 2 --output-dir ./outputs/sft
    """
    from torch.optim import AdamW

    from rlpost.sft_engine import TINY_SFT_ROWS, run_sft_training, tiny_sft_config
    from rlpost.tiny_lm import TinyCharTokenizer, tiny_causal_lm

    setup_logging(verbose)

    if not tiny:
        raise typer.BadParameter(
            "Pass --tiny. train-sft trains tiny_lm on CPU and does not download a model."
        )
    if output_dir is None:
        raise typer.BadParameter("--output-dir is required")
    if isinstance(steps, bool) or steps < 1:
        raise typer.BadParameter("--steps must be an int >= 1")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise typer.BadParameter("--batch-size must be an int >= 1")
    if resume and checkpoint is None:
        raise typer.BadParameter("--resume needs --checkpoint")
    run_seed = 42 if seed is None else seed
    if isinstance(run_seed, bool) or run_seed < 0:
        raise typer.BadParameter("--seed must be an int >= 0")

    try:
        train_config = tiny_sft_config(
            output_dir=output_dir,
            seed=run_seed,
            batch_size=batch_size,
            checkpoint_path=checkpoint,
        )
        model = tiny_causal_lm(train_config.seed)
        optimizer = AdamW(
            model.parameters(),
            lr=train_config.optimization.learning_rate,
            weight_decay=train_config.optimization.weight_decay,
        )
        result = run_sft_training(
            model,
            optimizer,
            train_config,
            steps,
            TINY_SFT_ROWS,
            TinyCharTokenizer(),
            resume_from=checkpoint if resume else None,
        )
        console.print(
            f"[green]train-sft finished[/green] step={result.step} "
            f"loss={result.losses[-1]:.6f}\n"
            f"  run: {result.output_path}"
        )
        if result.checkpoint_path is not None:
            console.print(f"  checkpoint: {result.checkpoint_path}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]", style="bold")
        raise typer.Exit(1)


@app.command()
def train_reward(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Directory for run.json. Required with --tiny.",
    ),
    tiny: bool = typer.Option(
        False,
        "--tiny",
        help="Train tiny_lm on CPU. Does not download a model.",
    ),
    steps: int = typer.Option(
        1,
        "--steps",
        help="Reward updates to take in this call. At least 1.",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        help="Pairs per update.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="TrainingConfig.seed. Default 42.",
    ),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint file written at the end of this call.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Load --checkpoint before the first step.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging",
    ),
) -> None:
    """Train a reward model on tiny_lm.

    ``--tiny`` builds one tiny sequence scorer and trains it on a fixed
    list of preference pairs. The loss is ``-log sigmoid(r_chosen -
    r_rejected)``. It does not download a model. Pass ``--checkpoint``
    to write a checkpoint at the end, and ``--resume`` to load that
    file first.

    Example:
        python -m rlpost.cli train-reward --tiny --steps 2 --output-dir ./outputs/reward
    """
    from torch.optim import AdamW

    from rlpost.reward_engine import TINY_REWARD_PAIRS, run_reward_training, tiny_reward_config
    from rlpost.tiny_lm import TinyCharTokenizer, tiny_reward_model

    setup_logging(verbose)

    if not tiny:
        raise typer.BadParameter(
            "Pass --tiny. train-reward trains tiny_lm on CPU and does not download a model."
        )
    if output_dir is None:
        raise typer.BadParameter("--output-dir is required")
    if isinstance(steps, bool) or steps < 1:
        raise typer.BadParameter("--steps must be an int >= 1")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise typer.BadParameter("--batch-size must be an int >= 1")
    if resume and checkpoint is None:
        raise typer.BadParameter("--resume needs --checkpoint")
    run_seed = 42 if seed is None else seed
    if isinstance(run_seed, bool) or run_seed < 0:
        raise typer.BadParameter("--seed must be an int >= 0")

    try:
        train_config = tiny_reward_config(
            output_dir=output_dir,
            seed=run_seed,
            batch_size=batch_size,
            checkpoint_path=checkpoint,
        )
        model = tiny_reward_model(train_config.seed)
        optimizer = AdamW(
            model.parameters(),
            lr=train_config.optimization.learning_rate,
            weight_decay=train_config.optimization.weight_decay,
        )
        result = run_reward_training(
            model,
            optimizer,
            train_config,
            steps,
            TINY_REWARD_PAIRS,
            TinyCharTokenizer(),
            resume_from=checkpoint if resume else None,
        )
        console.print(
            f"[green]train-reward finished[/green] step={result.step} "
            f"loss={result.losses[-1]:.6f}\n"
            f"  run: {result.output_path}"
        )
        if result.checkpoint_path is not None:
            console.print(f"  checkpoint: {result.checkpoint_path}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]", style="bold")
        raise typer.Exit(1)


@app.command()
def run_ppo(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Directory for run.json. Required with --tiny.",
    ),
    tiny: bool = typer.Option(
        False,
        "--tiny",
        help="Train tiny_lm on CPU. Does not download a model.",
    ),
    steps: int = typer.Option(
        1,
        "--steps",
        help="PPO updates to take in this call. At least 1.",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        help="Prompts per update.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="TrainingConfig.seed. Default 42.",
    ),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint file written at the end of this call.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Load --checkpoint before the first step.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging",
    ),
) -> None:
    """Train PPO on tiny_lm.

    ``--tiny`` builds one tiny Llama, a value head, and freezes a copy of
    the initial policy as the reference. It does not draw a second
    initialization. The prompts are a fixed list. The reward is the sum
    of token ids in the response window. Pass ``--checkpoint`` to write a
    checkpoint at the end, and ``--resume`` to load that file first. A
    Hugging Face model id is not loaded.

    Example:
        python -m rlpost.cli run-ppo --tiny --steps 2 --output-dir ./outputs/ppo
    """
    from torch.optim import AdamW

    from rlpost.dpo_engine import frozen_reference_copy
    from rlpost.ppo_engine import (
        TINY_PPO_PROMPTS,
        PPOTrainer,
        run_ppo_training,
        tiny_ppo_config,
    )
    from rlpost.tiny_lm import TinyCharTokenizer, tiny_causal_lm, tiny_value_model

    setup_logging(verbose)

    if not tiny:
        raise typer.BadParameter(
            "Pass --tiny. run-ppo trains tiny_lm on CPU and does not download a model."
        )
    if output_dir is None:
        raise typer.BadParameter("--output-dir is required")
    if isinstance(steps, bool) or steps < 1:
        raise typer.BadParameter("--steps must be an int >= 1")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise typer.BadParameter("--batch-size must be an int >= 1")
    if resume and checkpoint is None:
        raise typer.BadParameter("--resume needs --checkpoint")
    run_seed = 42 if seed is None else seed
    if isinstance(run_seed, bool) or run_seed < 0:
        raise typer.BadParameter("--seed must be an int >= 0")

    try:
        train_config = tiny_ppo_config(
            output_dir=output_dir,
            seed=run_seed,
            batch_size=batch_size,
            checkpoint_path=checkpoint,
        )
        policy = tiny_causal_lm(train_config.seed)
        value = tiny_value_model(train_config.seed)
        reference = frozen_reference_copy(policy)
        optimizer = AdamW(
            list(policy.parameters()) + list(value.parameters()),
            lr=train_config.optimization.learning_rate,
        )
        trainer = PPOTrainer(
            train_config,
            policy_model=policy,
            value_model=value,
            reference_model=reference,
            optimizer=optimizer,
        )
        result = run_ppo_training(
            trainer,
            steps,
            TINY_PPO_PROMPTS,
            TinyCharTokenizer(),
            resume_from=checkpoint if resume else None,
        )
        console.print(
            f"[green]run-ppo finished[/green] step={result.step} "
            f"loss={result.losses[-1]:.6f}\n"
            f"  run: {result.output_path}"
        )
        if result.checkpoint_path is not None:
            console.print(f"  checkpoint: {result.checkpoint_path}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]", style="bold")
        raise typer.Exit(1)


@app.command()
def run_dpo(
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Directory for run.json. Required with --tiny.",
    ),
    tiny: bool = typer.Option(
        False,
        "--tiny",
        help="Train tiny_lm on CPU. Does not download a model.",
    ),
    steps: int = typer.Option(
        1,
        "--steps",
        help="DPO updates to take in this call. At least 1.",
    ),
    batch_size: int = typer.Option(
        4,
        "--batch-size",
        help="Preference pairs per update.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="TrainingConfig.seed. Default 42.",
    ),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint file written at the end of this call.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Load --checkpoint before the first step.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging",
    ),
) -> None:
    """Train DPO on tiny_lm.

    ``--tiny`` builds one tiny Llama and freezes a copy of it as the
    reference. It does not draw a second initialization. The strings are
    a fixed list. Pass ``--checkpoint`` to write a checkpoint at the end,
    and ``--resume`` to load that file first. A Hugging Face model id is
    not loaded.

    Example:
        python -m rlpost.cli run-dpo --tiny --steps 2 --output-dir ./outputs/dpo
    """
    from rlpost.dpo_engine import (
        TINY_DPO_PAIRS,
        DPOTrainer,
        frozen_reference_copy,
        run_dpo_training,
        tiny_dpo_config,
    )
    from rlpost.tiny_lm import TinyCharTokenizer, tiny_causal_lm

    setup_logging(verbose)

    if not tiny:
        raise typer.BadParameter(
            "Pass --tiny. run-dpo trains tiny_lm on CPU and does not download a model."
        )
    if output_dir is None:
        raise typer.BadParameter("--output-dir is required")
    if isinstance(steps, bool) or steps < 1:
        raise typer.BadParameter("--steps must be an int >= 1")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise typer.BadParameter("--batch-size must be an int >= 1")
    if resume and checkpoint is None:
        raise typer.BadParameter("--resume needs --checkpoint")
    run_seed = 42 if seed is None else seed
    if isinstance(run_seed, bool) or run_seed < 0:
        raise typer.BadParameter("--seed must be an int >= 0")

    try:
        train_config = tiny_dpo_config(
            output_dir=output_dir,
            seed=run_seed,
            batch_size=batch_size,
            checkpoint_path=checkpoint,
        )
        policy = tiny_causal_lm(train_config.seed)
        reference = frozen_reference_copy(policy)
        trainer = DPOTrainer(
            train_config,
            policy_model=policy,
            reference_model=reference,
            tokenizer=TinyCharTokenizer(),
        )
        result = run_dpo_training(
            trainer,
            steps,
            TINY_DPO_PAIRS,
            resume_from=checkpoint if resume else None,
        )
        console.print(
            f"[green]run-dpo finished[/green] step={result.step} "
            f"loss={result.losses[-1]:.6f}\n"
            f"  run: {result.output_path}"
        )
        if result.checkpoint_path is not None:
            console.print(f"  checkpoint: {result.checkpoint_path}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]", style="bold")
        raise typer.Exit(1)


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        help="Show version",
    ),
) -> None:
    """SFT, a reward model, PPO, and DPO.

    For detailed help on any command:
        python -m rlpost.cli COMMAND --help
    """
    if version:
        console.print("rlpost version 0.1.0")
        raise typer.Exit()


if __name__ == "__main__":
    app()
