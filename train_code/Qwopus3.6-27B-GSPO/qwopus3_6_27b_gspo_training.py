# -*- coding: utf-8 -*-
# %% [markdown]
# # 0. Overview and safety notes
#
# This tutorial is a publication-safe GSPO-style post-training example for
# Jackrong/Qwopus3.6-27B-v2. It keeps the important training ideas from the
# private reference workflow while replacing local paths, private datasets, and
# project-specific values with generic placeholders.
#
# Safety defaults:
# - DRY_RUN is True.
# - START_TRAINING is False.
# - Heavy export and Hugging Face upload flags are False.
# - No token is hard-coded. Export HF_TOKEN and WANDB_API_KEY in your shell.
#
# The installed TRL API still exposes GRPOConfig and GRPOTrainer. This script
# keeps those class names in executable code because they are upstream API names,
# then configures sequence-level importance sampling and dr_grpo loss to match
# the GSPO-style behavior used by the reference workflow.

# %%
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ============================================================
# Shared prompt and answer tags
# ============================================================

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"
THINKING_COMPLETION_PREFIX = f"{THINK_START}\n"

ANSWER_INSTRUCTION = (
    "Reason through the problem step by step. Close the reasoning with "
    f"{THINK_END}, then give the final answer after it. Prefer wrapping only "
    f"the final answer in {ANSWER_START} and {ANSWER_END}. For multiple-choice "
    "questions, the answer tag should contain only the option letter."
)

_answer_tag_re = re.compile(
    rf"{re.escape(ANSWER_START)}\s*(.*?)\s*{re.escape(ANSWER_END)}",
    re.DOTALL | re.IGNORECASE,
)
_think_tag_re = re.compile(
    rf"{re.escape(THINK_START)}\s*(.*?)\s*{re.escape(THINK_END)}",
    re.DOTALL | re.IGNORECASE,
)
_boxed_re = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
_answer_colon_re = re.compile(
    r"(?i)(?:^|\n)\s*(?:final\s+)?answer\s*:\s*(?:\\boxed\{)?\s*([^}\n]+)",
    re.DOTALL,
)
_angle_answer_re = re.compile(r"<\s*([A-Za-z0-9.+\-/, ]{1,80})\s*>")
_mcqa_option_re = re.compile(
    r"(?:^|[\s\n\r])\(?([A-J])\)?\s*[\.\):]\s*(.*?)(?="
    r"(?:[\s\n\r]+\(?[A-J]\)?\s*[\.\):]\s*)|\Z)",
    re.DOTALL,
)


# %% [markdown]
# # 1. Imports and environment checks
#
# These helpers keep imports lightweight during dry runs. Expensive packages
# such as Unsloth and TRL are imported only inside the functions that need them.

# %%
def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def check_env() -> None:
    packages = [
        "torch",
        "unsloth",
        "unsloth_zoo",
        "trl",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "vllm",
        "peft",
        "tokenizers",
        "huggingface_hub",
    ]
    print(f"python: {sys.version.split()[0]}")
    for package in packages:
        print(f"{package}: {package_version(package)}")

    try:
        import torch
    except Exception as exc:
        print(f"torch import: ERROR {exc}")
        return

    print(f"torch cuda: {getattr(torch.version, 'cuda', None)}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gib = props.total_memory / (1024**3)
            print(f"cuda:{index}: {props.name}, total_memory={gib:.1f} GiB")


def explain_trl_import_error(exc: Exception) -> RuntimeError:
    return RuntimeError(
        "TRL GRPOConfig/GRPOTrainer could not be imported. The validation "
        "environment reported this when the optional vllm_ascend module was not "
        "installed. Install a compatible TRL/vLLM stack or remove the broken "
        "optional vLLM integration before constructing the trainer. The dry-run "
        "dataset and reward tests do not require this import."
    )


# %% [markdown]
# # 2. User-editable paths and model configuration
#
# Update these values for your own machine. The defaults are safe examples, not
# private server paths. A 27B-class multimodal model can require substantial
# VRAM or unified memory; reduce context length, LoRA rank, per-device batch
# size, and generation count before attempting a low-memory run.

# %%
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_NAME = os.getenv("MODEL_NAME", "Jackrong/Qwopus3.6-27B-v2")
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/workspace/example-user/qwopus-gspo-27b"))
DATASET_PATH = Path(
    os.getenv("DATASET_PATH", str(SCRIPT_DIR / "examples" / "sample_dataset_format.jsonl"))
)
OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", "/workspace/example-user/outputs/qwopus3.6-27b-gspo")
)
LORA_OUTPUT_DIR = Path(os.getenv("LORA_OUTPUT_DIR", str(OUTPUT_DIR) + "_lora"))
MERGED_16BIT_DIR = Path(
    os.getenv("MERGED_16BIT_DIR", str(OUTPUT_DIR) + "_merged_16bit")
)
GGUF_OUTPUT_DIR = Path(os.getenv("GGUF_OUTPUT_DIR", str(OUTPUT_DIR) + "_gguf"))

HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
HF_REPO_16BIT = os.getenv(
    "HF_REPO_16BIT", f"{HF_USERNAME}/Qwopus3.6-27B-v2-GSPO-16bit"
)
HF_REPO_GGUF = os.getenv(
    "HF_REPO_GGUF", f"{HF_USERNAME}/Qwopus3.6-27B-v2-GSPO-GGUF"
)
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "qwopus3.6-27b-gspo-public-example")

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Keep dry-run enabled while learning the file. Set START_TRAINING=True only
# after the dataset, rewards, memory budget, and output paths are reviewed.
DRY_RUN = env_bool("DRY_RUN", True)
START_TRAINING = env_bool("START_TRAINING", False)

# Heavy operations default to False so a full model export or upload is never
# triggered by accident.
EXPORT_LORA_ADAPTER = env_bool("EXPORT_LORA_ADAPTER", True)
EXPORT_MERGED_16BIT_LOCAL = env_bool("EXPORT_MERGED_16BIT_LOCAL", False)
PUSH_MERGED_16BIT_TO_HUB = env_bool("PUSH_MERGED_16BIT_TO_HUB", False)
EXPORT_GGUF_Q8_0_LOCAL = env_bool("EXPORT_GGUF_Q8_0_LOCAL", False)
PUSH_GGUF_Q8_0_TO_HUB = env_bool("PUSH_GGUF_Q8_0_TO_HUB", False)
EXPORT_MMPROJ_LOCAL = env_bool("EXPORT_MMPROJ_LOCAL", False)
RUN_GGUF_COMMANDS = env_bool("RUN_GGUF_COMMANDS", False)

# Use the built-in model code when possible. Set this to true only for model
# repos that require custom code and only after reviewing that code.
TRUST_REMOTE_CODE = env_bool("TRUST_REMOTE_CODE", False)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = MODEL_NAME

    # MAX_SEQ_LENGTH controls the total prompt plus completion context. Higher
    # values increase VRAM sharply; lower this first on memory-constrained GPUs.
    max_seq_length: int = env_int("MAX_SEQ_LENGTH", 4096)

    # 4-bit loading reduces memory use and matches the reference Unsloth LoRA
    # workflow. Disable only when you have enough memory for higher precision.
    load_in_4bit: bool = env_bool("LOAD_IN_4BIT", True)

    # Fast inference can improve generation speed, but some local stacks have
    # vLLM conflicts. Keep it off until the environment is verified.
    fast_inference: bool = env_bool("FAST_INFERENCE", False)

    # The reference workflow used Unsloth's qwen3-thinking template. The public
    # model also ships a Qwen3-VL processor chat template; leave this as "auto"
    # if your tokenizer already has the correct template.
    chat_template: str = os.getenv("CHAT_TEMPLATE", "auto")


@dataclass(frozen=True)
class LoraConfigPublic:
    # LORA_R controls adapter capacity. Higher rank can learn more but uses more
    # memory and may overfit small datasets.
    rank: int = env_int("LORA_R", 16)

    # LORA_ALPHA scales adapter updates. A common starting point is alpha=rank.
    alpha: int | None = None

    # LORA_DROPOUT may improve generalization on tiny datasets. Keep 0.0 for
    # the Unsloth reference-style setup unless you see overfitting.
    dropout: float = env_float("LORA_DROPOUT", 0.0)

    bias: str = "none"
    finetune_vision_layers: bool = env_bool("FINETUNE_VISION_LAYERS", False)
    finetune_language_layers: bool = env_bool("FINETUNE_LANGUAGE_LAYERS", True)
    finetune_attention_modules: bool = env_bool("FINETUNE_ATTENTION_MODULES", True)
    finetune_mlp_modules: bool = env_bool("FINETUNE_MLP_MODULES", True)

    # Gradient checkpointing trades compute for lower memory. The Unsloth value
    # is preserved from the reference script.
    use_gradient_checkpointing: str = os.getenv("USE_GRADIENT_CHECKPOINTING", "unsloth")

    use_rslora: bool = env_bool("USE_RSLORA", False)
    loftq_config: Any | None = None
    random_state: int = env_int("SEED", 3407)

    @property
    def resolved_alpha(self) -> int:
        return self.rank if self.alpha is None else self.alpha


@dataclass(frozen=True)
class DatasetConfig:
    dataset_path: Path = DATASET_PATH
    max_train_samples: int = env_int("MAX_TRAIN_SAMPLES", 128)
    sample_strategy: str = os.getenv("SAMPLE_STRATEGY", "first")
    seed: int = env_int("SEED", 3407)


@dataclass(frozen=True)
class TrainingConfig:
    # PER_DEVICE_TRAIN_BATCH_SIZE controls examples per optimizer micro-step on
    # each process. Lower it when you run out of memory.
    per_device_train_batch_size: int = env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1)

    # GRADIENT_ACCUMULATION_STEPS increases effective batch size with lower
    # micro-batch memory. Higher values slow each optimizer update.
    gradient_accumulation_steps: int = env_int("GRADIENT_ACCUMULATION_STEPS", 8)

    # NUM_GENERATIONS controls how many completions are sampled per prompt for
    # reward comparison. It increases generation workload and memory pressure,
    # but it is not multiplied into the standard optimizer batch-size formula.
    num_generations: int = env_int("NUM_GENERATIONS", 2)

    # MAX_PROMPT_LENGTH reserves tokens for the input prompt. Lower it if your
    # data is short or memory is tight.
    max_prompt_length: int = env_int("MAX_PROMPT_LENGTH", 1024)

    # MAX_COMPLETION_LENGTH reserves generated tokens. Lower this aggressively
    # when experimenting on smaller hardware.
    max_completion_length: int = env_int("MAX_COMPLETION_LENGTH", 3072)

    # LEARNING_RATE controls update size. Increase carefully for faster learning;
    # decrease when rewards are unstable or the model degrades.
    learning_rate: float = env_float("LEARNING_RATE", 5e-6)

    adam_beta1: float = env_float("ADAM_BETA1", 0.9)
    adam_beta2: float = env_float("ADAM_BETA2", 0.99)
    weight_decay: float = env_float("WEIGHT_DECAY", 0.1)

    # WARMUP_RATIO ramps the learning rate at the start. More warmup can improve
    # stability for noisy rewards, but it delays full-speed learning.
    warmup_ratio: float = env_float("WARMUP_RATIO", 0.1)

    lr_scheduler_type: str = os.getenv("LR_SCHEDULER_TYPE", "cosine")
    optim: str = os.getenv("OPTIM", "adamw_8bit")

    # MAX_STEPS is deliberately small for the tutorial. Raise it only after a
    # dry run and a short smoke run look correct.
    max_steps: int = env_int("MAX_STEPS", 10)

    # SAVE_STEPS controls checkpoint frequency. Lower values give more recovery
    # points but use more disk.
    save_steps: int = env_int("SAVE_STEPS", 5)

    # LOGGING_STEPS controls console/WandB frequency. Very low values are useful
    # for debugging but add overhead and noisy logs.
    logging_steps: int = env_int("LOGGING_STEPS", 1)

    max_grad_norm: float = env_float("MAX_GRAD_NORM", 0.1)
    report_to: str = os.getenv("REPORT_TO", "none")
    output_dir: Path = OUTPUT_DIR
    resume_from_checkpoint: str | None = os.getenv("RESUME_FROM_CHECKPOINT") or None

    # These fields are the public script's GSPO-style core. TRL still calls the
    # class GRPOConfig, but sequence-level importance sampling plus dr_grpo loss
    # preserves the reference workflow's GSPO behavior.
    importance_sampling_level: str = os.getenv("IMPORTANCE_SAMPLING_LEVEL", "sequence")
    loss_type: str = os.getenv("LOSS_TYPE", "dr_grpo")
    mask_truncated_completions: bool = env_bool("MASK_TRUNCATED_COMPLETIONS", True)


@dataclass(frozen=True)
class ExportConfig:
    lora_output_dir: Path = LORA_OUTPUT_DIR
    merged_16bit_dir: Path = MERGED_16BIT_DIR
    gguf_output_dir: Path = GGUF_OUTPUT_DIR
    llama_cpp_dir: Path = Path(os.getenv("LLAMA_CPP_DIR", "/workspace/llama.cpp"))


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    lora: LoraConfigPublic
    dataset: DatasetConfig
    training: TrainingConfig
    export: ExportConfig


def build_app_config() -> AppConfig:
    return AppConfig(
        model=ModelConfig(),
        lora=LoraConfigPublic(),
        dataset=DatasetConfig(),
        training=TrainingConfig(),
        export=ExportConfig(),
    )


def print_config(config: AppConfig) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    print(json.dumps(convert(asdict(config)), indent=2, ensure_ascii=False))
    print(
        "Effective optimizer batch size ~= "
        "per_device_train_batch_size x gradient_accumulation_steps x number_of_processes"
    )
    print(
        "Generation workload also scales with num_generations, but that is a "
        "sampling cost rather than the standard optimizer batch-size formula."
    )


# %% [markdown]
# # 3. Model loading
#
# This section preserves the reference workflow's Unsloth FastModel loader and
# LoRA attachment pattern. The model metadata for Jackrong/Qwopus3.6-27B-v2
# indicates a Qwen3-VL-style processor and vision configuration, so multimodal
# users should keep the processor files with any export.

# %%
def load_unsloth_model(model_config: ModelConfig, lora_config: LoraConfigPublic):
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_config.model_name,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        fast_inference=model_config.fast_inference,
    )

    if model_config.chat_template != "auto":
        tokenizer = get_chat_template(tokenizer, chat_template=model_config.chat_template)

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=lora_config.finetune_vision_layers,
        finetune_language_layers=lora_config.finetune_language_layers,
        finetune_attention_modules=lora_config.finetune_attention_modules,
        finetune_mlp_modules=lora_config.finetune_mlp_modules,
        r=lora_config.rank,
        lora_alpha=lora_config.resolved_alpha,
        lora_dropout=lora_config.dropout,
        bias=lora_config.bias,
        random_state=lora_config.random_state,
        use_rslora=lora_config.use_rslora,
        loftq_config=lora_config.loftq_config,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
    )
    return model, tokenizer


def check_model_config(model_name: str = MODEL_NAME) -> None:
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=TRUST_REMOTE_CODE)
    print(f"model_type: {getattr(config, 'model_type', None)}")
    print(f"architectures: {getattr(config, 'architectures', None)}")
    print(f"torch_dtype: {getattr(config, 'torch_dtype', None)}")
    print(f"has vision_config: {hasattr(config, 'vision_config')}")

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    print(f"processor/tokenizer class: {processor.__class__.__name__}")


# %% [markdown]
# # 4. Dataset loading and preprocessing
#
# The private reference consumed Responses-style JSONL rows containing
# `responses_create_params.input` and `expected_answer`. This public version also
# accepts a beginner-friendly schema with `messages` or `prompt`/`question`.

# %%
def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get("text") or item.get("content") or ""))
            else:
                texts.append(str(item))
        return "\n".join(text for text in texts if text)
    return "" if content is None else str(content)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                parts.append(message_content_to_text(content))
            else:
                parts.append(str(item))
        return "".join(parts)
    return "" if completion is None else str(completion)


def extract_response_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    params = example.get("responses_create_params") or {}
    messages: list[dict[str, str]] = []
    for item in params.get("input") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        content = message_content_to_text(item.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def extract_public_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(example.get("messages"), list):
        messages: list[dict[str, str]] = []
        for item in example["messages"]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user")
            content = message_content_to_text(item.get("content"))
            if content:
                messages.append({"role": role, "content": content})
        return messages

    prompt = example.get("prompt") or example.get("question") or example.get("instruction")
    if prompt:
        return [{"role": "user", "content": str(prompt)}]
    return []


def has_tools(example: dict[str, Any]) -> bool:
    params = example.get("responses_create_params") or {}
    return bool(params.get("tools"))


def is_trainable_answer_sample(example: dict[str, Any]) -> bool:
    if has_tools(example) or example.get("_hf_placeholder") is not None:
        return False
    if example.get("expected_answer") is None:
        return False
    messages = extract_response_messages(example) or extract_public_messages(example)
    return any(message["role"] == "user" for message in messages)


def summarize_prompt(prompt: list[dict[str, str]], limit: int = 700) -> str:
    user_messages = [message["content"] for message in prompt if message["role"] == "user"]
    text = "\n\n".join(user_messages) if user_messages else json.dumps(prompt, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def make_conversation(example: dict[str, Any]) -> dict[str, Any]:
    prompt = extract_response_messages(example) or extract_public_messages(example)
    if not prompt:
        raise ValueError("Each row needs Responses input, messages, prompt, question, or instruction.")
    if example.get("expected_answer") is None:
        raise ValueError("Each trainable row needs expected_answer.")

    for index in range(len(prompt) - 1, -1, -1):
        if prompt[index]["role"] == "user":
            prompt[index] = {
                **prompt[index],
                "content": f"{prompt[index]['content']}\n\n{ANSWER_INSTRUCTION}",
            }
            break
    else:
        prompt.append({"role": "user", "content": ANSWER_INSTRUCTION})

    return {
        "prompt": prompt,
        "expected_answer": str(example["expected_answer"]),
        "source_dataset": str(
            example.get("source_dataset") or example.get("dataset") or "public-example"
        ),
        "prompt_summary": summarize_prompt(prompt),
        "row_id": str(
            example.get("row_id")
            or example.get("uuid")
            or example.get("id")
            or example.get("hash_id")
            or example.get("question", "")[:80]
        ),
        "completion_prefix": THINKING_COMPLETION_PREFIX,
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
            yield line_number, loaded


def load_train_rows(config: DatasetConfig, scan_all: bool = False) -> list[dict[str, Any]]:
    path = config.dataset_path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Set DATASET_PATH or edit DatasetConfig."
        )

    rows: list[dict[str, Any]] = []
    rng = random.Random(config.seed)
    counts: Counter[str] = Counter()
    total_rows = 0
    trainable_rows = 0
    use_limit = config.max_train_samples > 0
    use_reservoir = config.sample_strategy == "reservoir"

    for _, example in iter_jsonl(path):
        total_rows += 1
        dataset_name = str(example.get("source_dataset") or example.get("dataset") or "")
        tools = has_tools(example)
        expected = example.get("expected_answer") is not None
        counts[f"{dataset_name}|tools={tools}|expected={expected}"] += 1

        if not is_trainable_answer_sample(example):
            continue

        trainable_rows += 1
        converted = make_conversation(example)
        if not use_limit or len(rows) < config.max_train_samples:
            rows.append(converted)
        elif use_reservoir:
            replacement_index = rng.randint(0, trainable_rows - 1)
            if replacement_index < config.max_train_samples:
                rows[replacement_index] = converted
        elif not scan_all:
            break

    rng.shuffle(rows)
    print(
        json.dumps(
            {
                "dataset_path": str(path),
                "total_rows_scanned": total_rows,
                "trainable_rows_seen": trainable_rows,
                "selected_rows": len(rows),
                "sample_strategy": config.sample_strategy,
                "scan_all": scan_all,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if scan_all:
        print("Top dataset buckets:")
        for key, count in counts.most_common(20):
            print(f"  {count:7d}  {key}")
    if not rows:
        raise RuntimeError("No trainable rows found after filtering.")
    return rows


def load_train_dataset(config: DatasetConfig, scan_all: bool = False):
    from datasets import Dataset

    return Dataset.from_list(load_train_rows(config, scan_all=scan_all))


def show_dataset_examples(config: DatasetConfig, scan_all: bool, show_rows: int) -> None:
    rows = load_train_rows(config, scan_all=scan_all)
    columns = sorted(rows[0].keys()) if rows else []
    print(f"Dataset columns: {columns}")
    for index, row in enumerate(rows[:show_rows]):
        print(
            json.dumps(
                {
                    "index": index,
                    "row_id": row["row_id"],
                    "source_dataset": row["source_dataset"],
                    "expected_answer": row["expected_answer"],
                    "prompt_summary": row["prompt_summary"][:1200],
                    "prompt": row["prompt"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


# %% [markdown]
# # 5. Prompt formatting and chat-template handling
#
# The trainer receives message lists so TRL can apply the processing class. For
# manual generation tests, this helper uses the tokenizer or processor chat
# template when one is available.

# %%
def format_prompt_for_generation(processing_class: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(processing_class, "apply_chat_template"):
        return processing_class.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    tokenizer = getattr(processing_class, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


# %% [markdown]
# # 6. Reward functions
#
# Rewards are passed separately to TRL. The trainer evaluates each function over
# the sampled completions and combines the reward signals internally. These
# functions preserve the reference behavior: formatting reward, anomaly penalty,
# and answer correctness reward.

# %%
def extract_tagged_answer(text: str) -> str | None:
    if THINK_START in text and THINK_END not in text:
        return None
    if THINK_END in text:
        text = text.rsplit(THINK_END, 1)[-1]

    matches = _answer_tag_re.findall(text)
    if matches:
        return matches[-1].strip()

    if ANSWER_START in text:
        tail = text.rsplit(ANSWER_START, 1)[-1]
        tail = tail.split(ANSWER_END, 1)[0]
        tail = re.split(r"[\n<]", tail, maxsplit=1)[0].strip()
        if 0 < len(tail) <= 100:
            return tail

    boxed = _boxed_re.findall(text)
    if boxed:
        return boxed[-1].strip()

    answer_colon = _answer_colon_re.findall(text)
    if answer_colon:
        return answer_colon[-1].strip()

    angle = _angle_answer_re.findall(text)
    if angle:
        for candidate in reversed(angle):
            cleaned = candidate.strip()
            if cleaned.startswith("/"):
                continue
            if cleaned.lower() in {"think", "answer", "reasoning", "solution"}:
                continue
            return cleaned

    return None


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if (
        (value.startswith("[") and value.endswith("]"))
        or (value.startswith("{") and value.endswith("}"))
    ):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list) and loaded:
                value = str(loaded[0])
            elif not isinstance(loaded, (list, dict)):
                value = str(loaded)
        except Exception:
            pass
    value = re.sub(r"\\boxed\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", value)
    value = value.replace("$", "").replace("\\", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .,\n\t")


def numeric_value(value: str) -> float | None:
    value = normalize_answer(value)
    value = value.replace(",", "").replace("%", "")
    value = re.sub(r"[^0-9eE+\-./]", "", value)
    if not value:
        return None
    try:
        if "/" in value and value.count("/") == 1:
            numerator, denominator = value.split("/")
            return float(numerator) / float(denominator)
        return float(value)
    except Exception:
        return None


def normalize_math(text: str) -> str:
    if not text:
        return ""
    text = text.replace("$", "").replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    while True:
        match = re.search(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
        if not match:
            break
        text = text.replace(match.group(0), f"{match.group(1)}/{match.group(2)}")
    text = text.replace("\\geqslant", ">=").replace("\\geq", ">=").replace("\\ge", ">=")
    text = text.replace("\\leqslant", "<=").replace("\\leq", "<=").replace("\\le", "<=")
    text = text.replace("\\neq", "!=").replace("\\ne", "!=")
    text = text.replace("\\cdot", "").replace("\\times", "")
    text = text.replace("\\pi", "pi").replace("\\infty", "infty")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("{", "").replace("}", "").replace("*", "")
    return text.lower()


def answers_match(predicted: str | None, expected: str) -> bool:
    if predicted is None:
        return False

    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    if not pred_norm or not exp_norm:
        return False

    if pred_norm.lower() == exp_norm.lower():
        return True

    if re.fullmatch(r"[A-Za-z]", exp_norm):
        cleaned_pred = re.sub(
            r"^(?:option|choice|select|answer|correct|is)\s+",
            "",
            pred_norm,
            flags=re.IGNORECASE,
        ).strip()
        if re.fullmatch(r"[A-Za-z]", cleaned_pred):
            return cleaned_pred.upper() == exp_norm.upper()
        return False

    pred_num = numeric_value(pred_norm)
    exp_num = numeric_value(exp_norm)
    if pred_num is not None and exp_num is not None:
        tolerance = max(1e-6, abs(exp_num) * 1e-6)
        return math.isclose(pred_num, exp_num, rel_tol=0, abs_tol=tolerance)

    if normalize_math(predicted) == normalize_math(expected):
        return True

    return False


def prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        return "\n".join(
            message_content_to_text(message.get("content"))
            for message in prompt
            if isinstance(message, dict)
        )
    return "" if prompt is None else str(prompt)


def extract_mcqa_options(prompt_text: str) -> dict[str, str]:
    options: dict[str, str] = {}
    compact = re.sub(r"\s+", " ", prompt_text).strip()
    for match in _mcqa_option_re.finditer(compact):
        letter = match.group(1).upper()
        value = match.group(2).strip()
        value = re.split(
            r"\s+(?:Use the model's normal|At the end of your response|"
            r"Your final answer|Make sure|Remember to|Please solve)",
            value,
            maxsplit=1,
        )[0].strip()
        if value:
            options[letter] = value
    return options


def answers_match_with_prompt(
    predicted: str | None,
    expected: str,
    prompt_text: str,
) -> bool:
    if answers_match(predicted, expected):
        return True
    if predicted is None:
        return False

    options = extract_mcqa_options(prompt_text)
    if not options:
        return False

    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    if re.fullmatch(r"[A-Za-z]", pred_norm) and not re.fullmatch(r"[A-Za-z]", exp_norm):
        option_value = options.get(pred_norm.upper())
        return answers_match(option_value, expected)

    if re.fullmatch(r"[A-Za-z]", exp_norm) and not re.fullmatch(r"[A-Za-z]", pred_norm):
        option_value = options.get(exp_norm.upper())
        return answers_match(predicted, option_value or "")

    return False


def completions_with_prefix(completions, completion_prefix=None) -> list[str]:
    if completion_prefix is None:
        prefixes = [""] * len(completions)
    elif isinstance(completion_prefix, str):
        prefixes = [completion_prefix] * len(completions)
    else:
        prefixes = ["" if value is None else str(value) for value in completion_prefix]
    return [
        f"{prefix}{completion_to_text(completion)}"
        for prefix, completion in zip(prefixes, completions)
    ]


def formatting_reward_func(completions, completion_prefix=None, **kwargs) -> list[float]:
    scores: list[float] = []
    for text in completions_with_prefix(completions, completion_prefix):
        score = 0.0
        answer_matches = _answer_tag_re.findall(text)
        think_matches = _think_tag_re.findall(text)
        extracted_answer = extract_tagged_answer(text)

        has_think_start = THINK_START in text
        has_think_end = THINK_END in text
        if has_think_start and has_think_end:
            score += 0.5
            final_region = text.rsplit(THINK_END, 1)[-1]
            if final_region.strip():
                score += 0.25
        elif has_think_start and not has_think_end:
            score -= 1.5

        if len(answer_matches) == 1 and answer_matches[0].strip():
            score += 1.0
        elif len(answer_matches) > 1:
            score -= 0.5
        elif extracted_answer is not None:
            score += 0.25

        if len(think_matches) > 1:
            score -= 0.25
        elif text.count(THINK_START) != text.count(THINK_END) and not (
            has_think_start and not has_think_end
        ):
            score -= 0.5

        if "<REASONING>" in text or "<SOLUTION>" in text:
            score -= 0.5
        scores.append(score)
    return scores


def anomaly_reward_func(completions, completion_prefix=None, **kwargs) -> list[float]:
    scores: list[float] = []
    for text in completions_with_prefix(completions, completion_prefix):
        score = 0.0
        if not text.strip():
            score -= 1.0
        if THINK_START in text and THINK_END not in text:
            score -= 1.5
        if text.count(ANSWER_START) > 1 or text.count(ANSWER_END) > 1:
            score -= 0.5
        if text.count(THINK_START) > 1 or text.count(THINK_END) > 1:
            score -= 0.25
        if len(text) > 0:
            removal = text.replace("addCriterion", "").replace("\n", "")
            if (len(text) - len(removal)) / max(len(text), 1) >= 0.5:
                score -= 2.0
        if extract_tagged_answer(text) is None:
            score -= 0.1
        scores.append(score)
    return scores


def correctness_reward_func(
    prompts,
    completions,
    expected_answer,
    source_dataset=None,
    prompt_summary=None,
    row_id=None,
    completion_prefix=None,
    **kwargs,
) -> list[float]:
    completions_text = completions_with_prefix(completions, completion_prefix)
    predicted_answers = [extract_tagged_answer(text) for text in completions_text]
    prompt_texts = [prompt_to_text(prompt) for prompt in prompts]
    return [
        2.0 if answers_match_with_prompt(predicted, expected, prompt_text) else 0.0
        for predicted, expected, prompt_text in zip(
            predicted_answers, expected_answer, prompt_texts
        )
    ]


def test_rewards() -> None:
    mcqa_value_prompt = (
        "Solve the expression. A. 6 B. 3 C. 146 D. 151.5 E. 16.5 "
        "Use the model's normal Qwen thinking format."
    )
    cases = [
        ("mcqa ok", f"{THINK_START}x{THINK_END}{ANSWER_START}F{ANSWER_END}", "F", True, "test"),
        ("mcqa boxed", "Reasoning...\nAnswer: \\boxed{G}", "G", True, "test"),
        ("mcqa option value", f"{THINK_START}x{THINK_END}{ANSWER_START}E{ANSWER_END}", "16.5", True, mcqa_value_prompt),
        ("angle answer", "After analysis the answer is <A>.", "A", True, "test"),
        ("numeric ok", f"{THINK_START}x{THINK_END}{ANSWER_START}149.0000001{ANSWER_END}", "149", True, "test"),
        ("fraction ok", f"{ANSWER_START}1/2{ANSWER_END}", "0.5", True, "test"),
        ("wrong", f"{THINK_START}x{THINK_END}{ANSWER_START}B{ANSWER_END}", "F", False, "test"),
        ("qwen thinking boxed", f"reasoning...{THINK_END}\n\nAnswer: \\boxed{{F}}", "F", True, "test"),
        ("unclosed think", f"{THINK_START}\nreasoning with Answer: \\boxed{{F}}", "F", False, "test"),
        ("empty", "", "F", False, "test"),
    ]
    completions = [case[1] for case in cases]
    expected = [case[2] for case in cases]
    correctness = correctness_reward_func(
        prompts=[[{"role": "user", "content": case[4]}] for case in cases],
        completions=completions,
        expected_answer=expected,
        source_dataset=["reward_unit_test"] * len(cases),
        prompt_summary=[case[4] for case in cases],
        row_id=[case[0] for case in cases],
    )
    formatting = formatting_reward_func(completions)
    formatting_with_qwen_prefix = formatting_reward_func(
        completions,
        completion_prefix=[THINKING_COMPLETION_PREFIX] * len(completions),
    )
    anomaly = anomaly_reward_func(completions)

    failures: list[str] = []
    for index, (name, text, expected_answer, should_match, prompt_text) in enumerate(cases):
        predicted = extract_tagged_answer(text)
        matched = answers_match_with_prompt(predicted, expected_answer, prompt_text)
        print(
            json.dumps(
                {
                    "case": name,
                    "expected": expected_answer,
                    "predicted": predicted,
                    "matched": matched,
                    "formatting_reward": formatting[index],
                    "formatting_with_qwen_prefix": formatting_with_qwen_prefix[index],
                    "anomaly_reward": anomaly[index],
                    "correctness_reward": correctness[index],
                },
                ensure_ascii=False,
            )
        )
        if matched != should_match:
            failures.append(name)
    if failures:
        raise AssertionError(f"Reward tests failed: {failures}")
    print("Reward tests passed.")


# %% [markdown]
# # 7. GSPO training hyperparameters
#
# Lower-memory GPU:
# - reduce context length first
# - reduce per-device batch size
# - reduce number of generated completions
# - reduce LoRA rank if needed
# - increase gradient accumulation steps to preserve effective batch size
#
# Higher-memory GPU:
# - increase context length only when the dataset needs it
# - increase the number of generations only when reward comparison quality benefits
# - scale effective batch size carefully

# %%
def build_training_args(config: TrainingConfig):
    try:
        from trl import GRPOConfig
    except Exception as exc:
        raise explain_trl_import_error(exc)

    return GRPOConfig(
        learning_rate=config.learning_rate,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        logging_steps=config.logging_steps,
        log_completions=False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_steps=config.max_steps,
        save_steps=config.save_steps,
        max_grad_norm=config.max_grad_norm,
        report_to=config.report_to,
        output_dir=str(config.output_dir),
        importance_sampling_level=config.importance_sampling_level,
        mask_truncated_completions=config.mask_truncated_completions,
        loss_type=config.loss_type,
        use_vllm=False,
    )


# %% [markdown]
# # 8. Trainer construction
#
# Constructing a real trainer loads the 27B model. Keep DRY_RUN enabled until the
# environment, data, and rewards are reviewed. The upstream class name remains
# GRPOTrainer because that is the TRL API name.

# %%
def build_trainer(config: AppConfig):
    if DRY_RUN:
        print("DRY_RUN=True, so build_trainer is not loading the 27B model.")
        print("Set DRY_RUN=False only after reviewing memory and paths.")
        return None, None, None

    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_unsloth_model(config.model, config.lora)
    train_dataset = load_train_dataset(config.dataset, scan_all=False)
    if len(train_dataset) == 0:
        raise RuntimeError("No trainable rows found.")

    print("First selected examples:")
    for index in range(min(5, len(train_dataset))):
        row = train_dataset[index]
        print(
            json.dumps(
                {
                    "index": index,
                    "source_dataset": row["source_dataset"],
                    "expected_answer": row["expected_answer"],
                    "prompt": row["prompt_summary"][:700],
                },
                ensure_ascii=False,
            )
        )

    try:
        from trl import GRPOTrainer
    except Exception as exc:
        raise explain_trl_import_error(exc)

    trainer = GRPOTrainer(
        model=model,
        args=build_training_args(config.training),
        processing_class=tokenizer,
        reward_funcs=[
            formatting_reward_func,
            anomaly_reward_func,
            correctness_reward_func,
        ],
        train_dataset=train_dataset,
    )
    return trainer, model, tokenizer


# %% [markdown]
# # 9. Start or resume training
#
# This section refuses to train unless both DRY_RUN is False and START_TRAINING
# is True. Use RESUME_FROM_CHECKPOINT to continue an interrupted run.

# %%
def start_or_resume_training(config: AppConfig):
    if DRY_RUN or not START_TRAINING:
        print("Training skipped. Set DRY_RUN=False and START_TRAINING=True to train.")
        return None, None, None

    trainer, model, tokenizer = build_trainer(config)
    if trainer is None:
        return None, None, None

    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    return trainer, model, tokenizer


# %% [markdown]
# # 10. Save LoRA adapter
#
# Adapter export is small compared with a merged model export, so it is enabled
# by default after a real training run.

# %%
def export_lora(model: Any, tokenizer: Any, config: ExportConfig) -> None:
    if not EXPORT_LORA_ADAPTER:
        print("LoRA adapter export skipped because EXPORT_LORA_ADAPTER=False.")
        return
    if model is None or tokenizer is None:
        print("No trained model/tokenizer was provided; LoRA adapter export skipped.")
        return
    config.lora_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.lora_output_dir))
    tokenizer.save_pretrained(str(config.lora_output_dir))
    print(f"Saved LoRA adapter to {config.lora_output_dir}")


# %% [markdown]
# # 11. Merge and export 16-bit model locally
#
# This is heavy for a 27B model and defaults to False. Unsloth models commonly
# support `save_pretrained_merged(..., save_method="merged_16bit")`.

# %%
def export_merged_16bit_local(model: Any, tokenizer: Any, config: ExportConfig) -> None:
    if not EXPORT_MERGED_16BIT_LOCAL:
        print("Merged 16-bit export skipped because EXPORT_MERGED_16BIT_LOCAL=False.")
        return
    if model is None or tokenizer is None:
        raise RuntimeError("A trained PEFT model and tokenizer are required for merge export.")

    config.merged_16bit_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(
            str(config.merged_16bit_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(config.merged_16bit_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(config.merged_16bit_dir))
    print(f"Saved merged 16-bit model to {config.merged_16bit_dir}")


# %% [markdown]
# # 12. Push merged 16-bit model to Hugging Face
#
# Uploads are disabled by default. Export HF_TOKEN in your shell instead of
# hard-coding credentials in Python.

# %%
def push_merged_16bit_to_hub(config: ExportConfig) -> None:
    if not PUSH_MERGED_16BIT_TO_HUB:
        print("16-bit Hub upload skipped because PUSH_MERGED_16BIT_TO_HUB=False.")
        return
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN in your shell before pushing to the Hub.")
    if not config.merged_16bit_dir.exists():
        raise FileNotFoundError(config.merged_16bit_dir)

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_REPO_16BIT, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=HF_REPO_16BIT,
        repo_type="model",
        folder_path=str(config.merged_16bit_dir),
        commit_message="Upload merged GSPO 16-bit model",
    )
    print(f"Pushed merged 16-bit model to {HF_REPO_16BIT}")


# %% [markdown]
# # 13. Export GGUF Q8_0 example locally
#
# Q8_0 is a near-full-precision-oriented GGUF example and produces larger files
# than lower-bit K-quants. This workflow assumes a merged local model exists.

# %%
def gguf_commands(config: ExportConfig) -> list[list[str]]:
    convert_script = config.llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_binary = config.llama_cpp_dir / "build" / "bin" / "llama-quantize"
    bf16_gguf = config.gguf_output_dir / "qwopus3.6-27b-gspo-bf16.gguf"
    q8_gguf = config.gguf_output_dir / "qwopus3.6-27b-gspo-Q8_0.gguf"
    return [
        [
            sys.executable,
            str(convert_script),
            str(config.merged_16bit_dir),
            "--outtype",
            "bf16",
            "--outfile",
            str(bf16_gguf),
        ],
        [str(quantize_binary), str(bf16_gguf), str(q8_gguf), "Q8_0"],
    ]


def export_gguf_q8_0_local(config: ExportConfig) -> None:
    if not EXPORT_GGUF_Q8_0_LOCAL:
        print("GGUF Q8_0 export skipped because EXPORT_GGUF_Q8_0_LOCAL=False.")
        return
    if not config.merged_16bit_dir.exists():
        raise FileNotFoundError(
            f"Merged model folder is required before GGUF export: {config.merged_16bit_dir}"
        )

    config.gguf_output_dir.mkdir(parents=True, exist_ok=True)
    commands = gguf_commands(config)
    print("GGUF commands:")
    for command in commands:
        print(" ".join(shlex.quote(part) for part in command))

    if not RUN_GGUF_COMMANDS:
        print("Set RUN_GGUF_COMMANDS=True after reviewing the commands to execute them.")
        return

    for command in commands:
        subprocess.run(command, check=True)


# %% [markdown]
# # 14. Push GGUF model to Hugging Face
#
# GGUF uploads are disabled by default. Review local files before enabling this.

# %%
def push_gguf_q8_0_to_hub(config: ExportConfig) -> None:
    if not PUSH_GGUF_Q8_0_TO_HUB:
        print("GGUF Hub upload skipped because PUSH_GGUF_Q8_0_TO_HUB=False.")
        return
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN in your shell before pushing to the Hub.")
    if not config.gguf_output_dir.exists():
        raise FileNotFoundError(config.gguf_output_dir)

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    api.create_repo(HF_REPO_GGUF, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=HF_REPO_GGUF,
        repo_type="model",
        folder_path=str(config.gguf_output_dir),
        commit_message="Upload GSPO GGUF Q8_0 model",
    )
    print(f"Pushed GGUF files to {HF_REPO_GGUF}")


# %% [markdown]
# # 15. List other supported GGUF quantization options
#
# The validation environment had llama.cpp with `llama-quantize` supporting
# F16, BF16, Q8_0, Q6_K, Q5_K_M, and Q4_K_M. The converter itself supported
# f32, f16, bf16, q8_0, tq1_0, tq2_0, and auto output types.

# %%
def list_supported_gguf_quantizations() -> None:
    options = [
        ("f16", "Half precision GGUF. Large; useful as a conversion source."),
        ("bf16", "BFloat16 GGUF. Large; useful when the source model is bfloat16."),
        ("q8_0", "Near-full-precision quantization; larger but simple baseline."),
        ("q6_k", "Smaller K-quant with moderate compression."),
        ("q5_k_m", "Common balanced K-quant for size and quality."),
        ("q4_k_m", "Smaller K-quant; useful for constrained inference hardware."),
    ]
    for name, description in options:
        print(f"{name}: {description}")


# %% [markdown]
# # 16. Export or preserve the multimodal projector (`mmproj`) when supported
#
# Jackrong/Qwopus3.6-27B-v2 advertises a Qwen3-VL processor and vision config,
# but no prebuilt mmproj file was listed during validation. The checked llama.cpp
# exposes an experimental `convert_hf_to_gguf.py --mmproj` option. Keep this
# disabled until you confirm the converter supports the exact architecture.

# %%
def inspect_mmproj_support(model_name: str = MODEL_NAME, export_config: ExportConfig | None = None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(model_name, files_metadata=False)
    names = [sibling.rfilename for sibling in info.siblings or []]
    projector_files = [
        name for name in names if "mmproj" in name.lower() or "projector" in name.lower()
    ]
    print(f"pipeline_tag: {info.pipeline_tag}")
    print(f"prebuilt projector files: {projector_files or 'none listed'}")
    print("If a projector file appears later, preserve it beside the GGUF model.")
    print(
        "For llama.cpp server usage, the conceptual load command is: "
        "llama-server -m path/to/model.gguf --mmproj path/to/mmproj-file.gguf"
    )
    if export_config is not None:
        convert_script = export_config.llama_cpp_dir / "convert_hf_to_gguf.py"
        command = [
            sys.executable,
            str(convert_script),
            str(export_config.merged_16bit_dir),
            "--mmproj",
            "--outfile",
            str(export_config.gguf_output_dir / "mmproj-qwopus3.6-27b-gspo.gguf"),
        ]
        print("Reviewed experimental mmproj export command:")
        print(" ".join(shlex.quote(part) for part in command))
        if EXPORT_MMPROJ_LOCAL and RUN_GGUF_COMMANDS:
            subprocess.run(command, check=True)
        else:
            print("mmproj export skipped by default.")


# %% [markdown]
# # 17. Smoke-test checklist
#
# Safe checks:
# - imports and package versions
# - CUDA visibility
# - dataset schema validation
# - reward functions on synthetic examples
# - optional model config and processor load
# - optional trainer construction only after DRY_RUN=False

# %%
def smoke_all(config: AppConfig) -> None:
    print_config(config)
    check_env()
    show_dataset_examples(config.dataset, scan_all=False, show_rows=3)
    test_rewards()
    config.export.lora_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Adapter output path exists or was created: {config.export.lora_output_dir}")
    print("Smoke checks finished without starting training.")


# %% [markdown]
# # 18. Troubleshooting notes
#
# Common OOM fixes, in priority order:
# 1. Lower MAX_SEQ_LENGTH.
# 2. Lower MAX_COMPLETION_LENGTH.
# 3. Lower PER_DEVICE_TRAIN_BATCH_SIZE.
# 4. Lower NUM_GENERATIONS.
# 5. Lower LORA_R.
# 6. Increase GRADIENT_ACCUMULATION_STEPS to recover effective batch size.
#
# Common environment issues:
# - TRL may import optional vLLM integrations even when `use_vllm=False`.
# - Package APIs evolve; compare installed versions against requirements-gspo.txt.
# - GGUF conversion requires a built llama.cpp checkout and enough disk space.
# - Multimodal GGUF support depends on current llama.cpp architecture support.

# %%
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public Qwopus3.6 27B GSPO tutorial")
    parser.add_argument(
        "command",
        nargs="?",
        default="smoke-all",
        choices=[
            "check-env",
            "check-model-config",
            "dry-run-data",
            "test-rewards",
            "dry-run-trainer",
            "train",
            "export-16bit",
            "push-16bit",
            "export-gguf-q8",
            "push-gguf",
            "list-gguf-quants",
            "check-mmproj",
            "smoke-all",
        ],
    )
    parser.add_argument("--scan-all", action="store_true")
    parser.add_argument("--show-rows", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_app_config()

    if args.command == "check-env":
        check_env()
    elif args.command == "check-model-config":
        check_model_config(config.model.model_name)
    elif args.command == "dry-run-data":
        show_dataset_examples(config.dataset, scan_all=args.scan_all, show_rows=args.show_rows)
    elif args.command == "test-rewards":
        test_rewards()
    elif args.command == "dry-run-trainer":
        build_trainer(config)
    elif args.command == "train":
        _, model, tokenizer = start_or_resume_training(config)
        export_lora(model, tokenizer, config.export)
        export_merged_16bit_local(model, tokenizer, config.export)
    elif args.command == "export-16bit":
        raise SystemExit(
            "export-16bit requires the live trained model object. Run `train` with "
            "EXPORT_MERGED_16BIT_LOCAL=True, or call export_merged_16bit_local(...) "
            "from an interactive session immediately after training."
        )
    elif args.command == "push-16bit":
        push_merged_16bit_to_hub(config.export)
    elif args.command == "export-gguf-q8":
        export_gguf_q8_0_local(config.export)
    elif args.command == "push-gguf":
        push_gguf_q8_0_to_hub(config.export)
    elif args.command == "list-gguf-quants":
        list_supported_gguf_quantizations()
    elif args.command == "check-mmproj":
        inspect_mmproj_support(config.model.model_name, config.export)
    elif args.command == "smoke-all":
        smoke_all(config)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
