# Qwopus3.6 27B GSPO Training Tutorial

This folder contains a beginner-friendly, publication-safe GSPO-style
post-training example for `Jackrong/Qwopus3.6-27B-v2`.

The tutorial is designed to teach the moving parts of a reward-based LoRA
training workflow: model loading, dataset conversion, prompt formatting, reward
functions, GSPO-style trainer settings, adapter saving, merged-model export,
GGUF conversion, and multimodal projector handling.

## Hardware Expectations

`Jackrong/Qwopus3.6-27B-v2` is a 27B-class multimodal model. Even with 4-bit
loading and LoRA, GSPO-style training samples multiple completions per prompt
and can require substantial VRAM or unified memory. Start with the small default
settings, validate the rewards, then scale carefully.

For lower-memory hardware, reduce settings in this order:

1. `MAX_SEQ_LENGTH`
2. `MAX_COMPLETION_LENGTH`
3. `PER_DEVICE_TRAIN_BATCH_SIZE`
4. `NUM_GENERATIONS`
5. `LORA_R`

Increase `GRADIENT_ACCUMULATION_STEPS` if you need a larger effective optimizer
batch size without increasing micro-batch memory.

## Folder Structure

```text
public_gspo_qwopus3.6_27b_example/
├── qwopus3_6_27b_gspo_training.py
├── ENVIRONMENT_SETUP.txt
├── README.md
├── SOURCE_PARITY_AUDIT.md
├── PRIVACY_AUDIT.md
├── requirements-gspo.txt
└── examples/
    └── sample_dataset_format.jsonl
```

## Setup

Create a virtual environment, install a CUDA-compatible `torch`, then install:

```bash
pip install -r requirements-gspo.txt
```

See `ENVIRONMENT_SETUP.txt` for operating system assumptions, CUDA notes,
optional WandB setup, Hugging Face login guidance, and GGUF conversion setup.

## Dataset Schema

The script accepts two schemas.

Beginner-friendly public schema:

```json
{
  "row_id": "sample-math-001",
  "source_dataset": "synthetic-public-example",
  "messages": [{"role": "user", "content": "Compute 12 * 7."}],
  "expected_answer": "84",
  "metadata": {"topic": "arithmetic"}
}
```

Minimal prompt schema:

```json
{"prompt": "Which option is prime? A. 21 B. 29", "expected_answer": "B"}
```

Responses-style schema:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "Question text"}]
  },
  "expected_answer": "final answer"
}
```

Rows with tool definitions or missing `expected_answer` are skipped, matching
the reference workflow's answer-only training filter.

## Reward Functions

The tutorial preserves three reward signals:

- `formatting_reward_func`: rewards completions that close `<think>` and place
  the final answer in exactly one `<answer>...</answer>` block.
- `anomaly_reward_func`: penalizes empty output, unclosed thinking tags,
  repeated answer tags, repeated thinking tags, and obvious malformed repetition.
- `correctness_reward_func`: extracts the final answer and compares it against
  `expected_answer`, including exact text, option-letter, numeric, fraction, and
  normalized math-expression matches.

TRL receives these functions as a list. The upstream trainer combines reward
signals while optimizing over sampled completions.

## GSPO Terminology

The installed TRL API still names the public classes `GRPOConfig` and
`GRPOTrainer`. This tutorial keeps those executable API names but configures the
workflow with the GSPO-style settings preserved from the reference:

```python
importance_sampling_level = "sequence"
loss_type = "dr_grpo"
mask_truncated_completions = True
num_generations = 2
```

If your installed TRL version changes these field names, inspect the local TRL
docs or source before editing the script.

## Hyperparameter Tuning

Important defaults are intentionally conservative:

```bash
MAX_SEQ_LENGTH=4096
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=3072
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
NUM_GENERATIONS=2
LEARNING_RATE=5e-6
MAX_STEPS=10
SAVE_STEPS=5
LORA_R=16
```

Effective optimizer batch size is approximately:

```text
per_device_train_batch_size x gradient_accumulation_steps x number_of_processes
```

`NUM_GENERATIONS` increases completion sampling workload for reward comparison.
It is not part of the standard optimizer batch-size formula.

## Safe Smoke Tests

Run these before training:

```bash
python qwopus3_6_27b_gspo_training.py check-env
python qwopus3_6_27b_gspo_training.py dry-run-data
python qwopus3_6_27b_gspo_training.py test-rewards
python qwopus3_6_27b_gspo_training.py list-gguf-quants
```

The default command is safe:

```bash
python qwopus3_6_27b_gspo_training.py
```

It does not start training.

## Train from Scratch

Review your data and memory budget first. Then:

```bash
export DATASET_PATH="/workspace/example-user/datasets/gspo_train.jsonl"
export OUTPUT_DIR="/workspace/example-user/outputs/qwopus3.6-27b-gspo"
export DRY_RUN=False
export START_TRAINING=True
python qwopus3_6_27b_gspo_training.py train
```

## Resume from a Checkpoint

```bash
export RESUME_FROM_CHECKPOINT="/workspace/example-user/outputs/qwopus3.6-27b-gspo/checkpoint-10"
export DRY_RUN=False
export START_TRAINING=True
python qwopus3_6_27b_gspo_training.py train
```

## Save LoRA Adapters

After a real training run, `EXPORT_LORA_ADAPTER=True` saves the LoRA adapter and
tokenizer files to `LORA_OUTPUT_DIR`. This is much smaller than a merged model.

## Merge and Save 16-Bit Model Locally

Merged export is disabled by default:

```bash
export EXPORT_MERGED_16BIT_LOCAL=True
python qwopus3_6_27b_gspo_training.py train
```

The script runs merged export after a real training run, while the trained model
object is still in memory. The standalone `export-16bit` command intentionally
exits with guidance because it does not reload a PEFT model from disk. The merge
uses Unsloth's `save_pretrained_merged(..., save_method="merged_16bit")` when
available, otherwise it falls back to PEFT's `merge_and_unload()` path.

## Push Merged 16-Bit Model to Hugging Face

Hub upload is disabled by default. Do not hard-code tokens:

```bash
export HF_TOKEN="your_huggingface_token"
export PUSH_MERGED_16BIT_TO_HUB=True
python qwopus3_6_27b_gspo_training.py push-16bit
```

## Export GGUF Q8_0

Q8_0 is a near-full-precision-oriented GGUF format with a larger file size than
lower-bit K-quants. Review commands first:

```bash
export EXPORT_GGUF_Q8_0_LOCAL=True
python qwopus3_6_27b_gspo_training.py export-gguf-q8
```

Run the commands only after review:

```bash
export RUN_GGUF_COMMANDS=True
python qwopus3_6_27b_gspo_training.py export-gguf-q8
```

## Push GGUF to Hugging Face

Disabled by default:

```bash
export HF_TOKEN="your_huggingface_token"
export PUSH_GGUF_Q8_0_TO_HUB=True
python qwopus3_6_27b_gspo_training.py push-gguf
```

## Other GGUF Quantization Formats

The validated llama.cpp tooling exposed these relevant formats:

- `f16`
- `bf16`
- `q8_0`
- `q6_k`
- `q5_k_m`
- `q4_k_m`

Use:

```bash
python qwopus3_6_27b_gspo_training.py list-gguf-quants
```

## Multimodal Projector (`mmproj`)

Model metadata indicates that `Jackrong/Qwopus3.6-27B-v2` is multimodal and
uses a Qwen3-VL processor. The model repository did not list a prebuilt
`mmproj` file during validation. The validated llama.cpp converter exposed an
experimental `--mmproj` option.

Check current support:

```bash
python qwopus3_6_27b_gspo_training.py check-mmproj
```

If a valid projector file is created or provided by the model repo, keep it
beside the GGUF model. Conceptual llama.cpp server usage:

```bash
llama-server -m path/to/model.gguf --mmproj path/to/mmproj-file.gguf
```

Do not create a fake `mmproj` file. If the converter cannot export it for this
architecture, document that limitation and keep processor files with the model.

## Privacy Notice

All paths, usernames, repositories, and sample data in this folder are
placeholders or public model identifiers. No private dataset rows, secrets,
tokens, server aliases, or local machine paths are intentionally included.

## Troubleshooting

- If TRL fails while importing an optional vLLM integration, install a compatible
  vLLM stack or use a TRL/vLLM combination that does not require that optional
  module.
- If CUDA OOM occurs, reduce context length, completion length, batch size,
  generation count, and LoRA rank in that order.
- If GGUF conversion fails, update llama.cpp and confirm the merged model folder
  contains config, tokenizer or processor files, and model weights.
- If the tokenizer or processor behavior changes, inspect the model repository's
  `tokenizer_config.json` and `processor_config.json`.

## License and Attribution

Add the final project license before publishing derived training outputs. This
tutorial references `Jackrong/Qwopus3.6-27B-v2` and relies on the licenses of the
model, training libraries, and any datasets you choose to use.
