# eval/compare_models.py
"""
compare_models.py

Main entrypoint for running evaluations in llm-eval-pro.

Usage (from project root):

    python -m eval.compare_models --config configs/eval_config.yaml

This will:
- load the dataset
- call one or more models
- compute metrics
- write per-example and aggregated CSVs under outputs/
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

from .metrics import evaluate_one, aggregate


# ---- Config & model definitions --------------------------------------------


@dataclass
class ModelConfig:
    name: str
    provider: str
    params: Dict[str, Any]


@dataclass
class EvalConfig:
    dataset_path: Path
    output_dir: Path
    models: List[ModelConfig]
    metrics: List[str]
    max_examples: int


def _project_root() -> Path:
    """
    Resolve project root as the parent of the 'eval' directory.
    This makes the script work regardless of current working directory,
    as long as it's called via `python -m eval.compare_models`.
    """
    return Path(__file__).resolve().parents[1]


def load_config(path: Path) -> EvalConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dataset_path = _project_root() / raw["dataset_path"]
    output_dir = _project_root() / raw["output_dir"]

    models = [
        ModelConfig(
            name=m["name"],
            provider=m["provider"],
            params=m.get("params", {}),
        )
        for m in raw["models"]
    ]

    metrics = list(raw["metrics"])
    max_examples = int(raw.get("max_examples", 100))

    return EvalConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        models=models,
        metrics=metrics,
        max_examples=max_examples,
    )


def load_dataset(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of records.")

    if not data:
        raise ValueError("Dataset is empty; nothing to evaluate.")

    if limit is not None:
        data = data[:limit]

    required_keys = {"id", "input", "expected"}
    for row in data:
        missing = required_keys - set(row.keys())
        if missing:
            raise ValueError(f"Dataset row missing keys: {missing}")

    return data


# ---- Model provider implementations ----------------------------------------


def _dummy_echo_model(prompt: str, *, prefix: str = "") -> str:
    """
    Simple built-in provider that echoes the user question.
    Useful to test the pipeline end-to-end.
    """
    return f"{prefix}{prompt}"


def _openai_chat_model(prompt: str, *, model_id: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
    """
    Example OpenAI call. This is written so it's easy to extend or replace.
    Requires:
        - `openai` installed
        - OPENAI_API_KEY in environment
    """
    try:
        import openai  # type: ignore
    except ImportError as exc:  # pragma: no cover - only triggered if missing
        raise RuntimeError(
            "openai package not installed. Run `pip install openai` "
            "or remove OpenAI models from your config."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    openai.api_key = api_key

    # NOTE: Depending on SDK version, you may need to adjust this call.
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for short factual QA."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def generate_answer(
    model: ModelConfig,
    *,
    question: str,
) -> str:
    """
    Dispatch to a specific provider implementation.

    In production you might split each provider into its own module,
    but for this repo we keep it in one place for simplicity.
    """
    provider = model.provider.lower()

    if provider == "dummy":
        prefix = str(model.params.get("prefix", ""))
        return _dummy_echo_model(question, prefix=prefix)

    if provider == "openai":
        model_id = model.params.get("model_id") or model.params.get("model") or model.params.get("model_name")
        if not model_id:
            # allow `model_id` to be set at the top level as well
            model_id = model.params.get("model_id") or getattr(model, "model_id", None)
        if not model_id:
            raise ValueError(f"OpenAI provider for {model.name} requires `model_id` in params.")

        temperature = float(model.params.get("temperature", 0.0))
        max_tokens = int(model.params.get("max_tokens", 256))
        return _openai_chat_model(
            prompt=question,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown provider: {model.provider}")


# ---- Evaluation loop --------------------------------------------------------


def _build_prompt(question: str) -> str:
    """
    Simple prompt builder. In a production setup you might inject
    system prompts, context, or templates here.
    """
    return f"Answer the following question concisely and factually.\n\nQuestion: {question}\nAnswer:"


def run_evaluation(config_path: Path) -> None:
    cfg = load_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(cfg.dataset_path, limit=cfg.max_examples)

    per_example_rows: List[Dict[str, Any]] = []
    aggregated_rows: List[Dict[str, Any]] = []

    for model_cfg in cfg.models:
        print(f"\n▶ Evaluating model: {model_cfg.name} (provider={model_cfg.provider})")
        per_example_metrics: List[Dict[str, float]] = []

        for row in dataset:
            q = row["input"]
            expected = row["expected"]

            prompt = _build_prompt(q)
            prediction = generate_answer(model_cfg, question=prompt)

            metrics = evaluate_one(prediction, expected, cfg.metrics)
            per_example_metrics.append(metrics)

            per_example_rows.append(
                {
                    "model": model_cfg.name,
                    "example_id": row["id"],
                    "input": q,
                    "expected": expected,
                    "prediction": prediction,
                    **metrics,
                }
            )

        agg = aggregate(per_example_metrics)
        aggregated_rows.append({"model": model_cfg.name, **agg})

    # Write outputs
    per_example_df = pd.DataFrame(per_example_rows)
    per_example_path = cfg.output_dir / "per_example_metrics.csv"
    per_example_df.to_csv(per_example_path, index=False)
    print(f"✅ Saved per-example metrics to {per_example_path}")

    aggregated_df = pd.DataFrame(aggregated_rows)
    aggregated_path = cfg.output_dir / "aggregated_metrics.csv"
    aggregated_df.to_csv(aggregated_path, index=False)
    print(f"✅ Saved aggregated metrics to {aggregated_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM evaluations with llm-eval-pro.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to eval_config.yaml (relative to project root).",
    )
    args = parser.parse_args()

    config_path = _project_root() / args.config
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    run_evaluation(config_path)


if __name__ == "__main__":
    main()
