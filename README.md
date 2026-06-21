# Safety-Efficiency Trade-offs: Evaluating the Impact of Prompt Compression on LLM Robustness Against Adversarial Attacks

> **JCC** — Companion repository with the full experimental pipeline, raw results, and figure generation scripts.

## Models Used

| Role | Model | Parameters | Quantization |
|------|-------|-----------:|:------------:|
| **Target LLM (victim)** | Meta-Llama-4-Scout | 109B (Sparse MoE) | Q4_K_M (GGUF) |
| **Target LLM (victim)** | Qwen-1.5-110B-Chat | 110B (Dense) | Q4_K_M (GGUF) |
| **Target LLM (victim)** | Gemma-2-27B-it | 27B (Dense) | Q4_K_M (GGUF) |
| **Primary Judge** | Llama-Guard-3-8B | 8B | Q4_K_M (GGUF) |
| **Extractive Compressor** | LLMLingua-2 (BERT-based) | 178M | FP32 (CPU) |
| **Abstractive Compressor** | BART-large-cnn | 406M | FP32 (CPU) |
| **Optional Detector** | Prompt-Guard-86M | 86M | FP32 (CPU) |

## Research Objective

This project implements an automated, end-to-end experimental pipeline to study how **extractive** (LLMLingua-2) and **abstractive** (BART) prompt compression affect the safety alignment of frontier Large Language Models (LLMs) against jailbreak attacks.

Key findings:
- **Extractive Vulnerability Spike**: Moderate extractive compression (LLMLingua-2) can *amplify* attacks by stripping low-entropy safety preambles while preserving high-entropy adversarial payloads.
- **Abstractive Neutralization Effect**: Abstractive compression (BART) systematically *neutralizes* attacks by collapsing adversarial camouflage into raw intent, triggering native safety filters.
- **Model-Dependent Heterogeneity**: Adversarial resilience is determined by the interaction between compression method, compression level, and the specific model's safety profile — not by architecture type (Dense vs. MoE).

## Research Question

> How does standard, unmodified prompt compression natively interact with the alignment of frontier-scale LLMs under standardized adversarial benchmarking?

## What This Pipeline Does

For each attack, the pipeline compares two conditions:
1. The original prompt, uncompressed.
2. The same prompt after compression with LLMLingua-2 (extractive) or BART (abstractive).

Both versions are sent to the target model, then an automated judge classifies each response as `safe` or `unsafe`. Metrics such as ASR, latency, tokens, and paired statistical tests are computed.

## General Flow

Processing occurs **sample by sample** (no global pre-batching).

For each dataset, each compression rate, and each prompt:

1. A malicious prompt is taken from the dataset.
2. Optionally evaluated with Prompt Guard before compression.
3. The prompt is compressed with LLMLingua-2 / BART.
4. Optionally re-evaluated with Prompt Guard after compression.
5. The target model responds to the original prompt.
6. The target model responds to the compressed prompt.
7. The judge classifies both responses as `safe` or `unsafe`.
8. Results, latencies, tokens, and metadata are saved.

This cycle repeats for all samples and all configured rates.

## Experimental Architecture

```text
Original prompt
      |
      v
[Optional] Prompt Guard
      |
      v
LLMLingua-2 / BART  --->  Compressed prompt
      |
      v
Target LLM
      |
      v
Automated Judge (Llama-Guard-3-8B)
      |
      v
Metrics: ASR / delta ASR / latency / tokens / detection
```

## Project Components

### 1. Compressor

Two compression paradigms are supported:

- **Extractive (LLMLingua-2)**: Token-level classification that prunes redundant tokens.
  - Model: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
  - Implementation: [pipeline/config.py](pipeline/config.py), [pipeline/run_experiment.py](pipeline/run_experiment.py)

- **Abstractive (BART-large-cnn)**: Sequence-to-sequence summarization that generates a condensed version.
  - Model: `facebook/bart-large-cnn`
  - Implementation: [pipeline/run_experiment.py](pipeline/run_experiment.py)
  - Activated via: `export COMPRESSOR_TYPE=abstractive`

### 2. Target Model

The target LLM under attack. The pipeline supports two modes:

1. `local`: loads Hugging Face models locally with Transformers.
2. `api`: uses an OpenAI-compatible endpoint (e.g., Ollama, Together).

Implementation: [pipeline/llm_client.py](pipeline/llm_client.py)

Models evaluated in the paper:
- Meta-Llama-4-Scout (109B, Sparse MoE)
- Qwen-1.5-110B-Chat (110B, Dense)
- Gemma-2-27B-it (27B, Dense)

### 3. Automated Judge

The judge determines whether the model's response is safe or unsafe.

Implementation: [pipeline/judge.py](pipeline/judge.py)

Primary judge: Llama-Guard-3-8B (local or API mode).

### 4. Prompt Guard

Optional injection/jailbreak detector for studying evasion rates.

Implementation: [pipeline/prompt_guard.py](pipeline/prompt_guard.py)

Provides two extra metrics:
1. Whether the attack was detectable before compression.
2. Whether it becomes undetectable after compression.

### 5. Metrics

Implementation: [pipeline/metrics.py](pipeline/metrics.py)

The system computes:
1. ASR without compression
2. ASR with compression
3. Delta ASR
4. Average latency
5. Average tokens
6. Average compression ratio
7. Exact McNemar test for paired comparison
8. Per-dataset metrics
9. Per-dataset and per-rate metrics

## Project Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── pipeline/                      # Experimental pipeline
│   ├── config.py
│   ├── llm_client.py
│   ├── judge.py
│   ├── metrics.py
│   ├── prompt_guard.py
│   ├── run_experiment.py
│   ├── aggregate_results.py       # Aggregates results for dashboard
│   └── fix_results.py             # Post-processing corrections
├── paper/                         # Paper source (LaTeX + figures)
│   ├── main.tex
│   ├── generate_figures.py
│   └── fig*.pdf / fig*.png        # Generated figures
├── results_final/                 # Paper results, organized by compressor
│   ├── llmlingua2/                #   Extractive compression (LLMLingua-2)
│   │   ├── llama4_scout/
│   │   ├── qwen_110b/
│   │   └── gemma2_27b/
│   ├── bart/                      #   Abstractive compression (BART)
│   │   ├── llama4_scout/
│   │   ├── qwen_110b/
│   │   └── gemma2_27b/
│   └── README.md
├── results_archive/               # Archived supplementary experiments
│   ├── resultados_server/         #   Early server runs
│   ├── otros_modelos/             #   Pilot experiments (smaller models)
│   └── README.md
├── dashboard/                     # Interactive visualization dashboard
└── experiments/                   # Run scripts and configuration
    ├── scripts/                   #   Experiment launch scripts
    ├── setup/                     #   Dependency installation
    ├── archive_scripts/           #   Early test scripts
    └── logs/                      #   Execution logs
```

## Configuration

Central configuration: [pipeline/config.py](pipeline/config.py)

### Profiles

- `PILOT`: Quick validation on a laptop (small models, few samples).
- `LAPTOP`: Hybrid mode with local compression + API-based LLM.
- `SERVER`: Full experiment on a server with Ollama (128GB RAM, no GPU required).
- `LAB`: Heavy run for the final experiment.

### Backend

- `local`: Transformers with Hugging Face models.
- `api`: OpenAI-compatible endpoint (Ollama default: `http://localhost:11434/v1`).

### Datasets

Defined in `DATASET_REGISTRY` in [pipeline/config.py](pipeline/config.py):
1. `jbb` — JailbreakBench/JBB-Behaviors (115 harmful intents)
2. `advbench` — walledai/AdvBench (adversarial templates)

### Compression Rates

The pipeline supports rate sweeping: `0.9, 0.7, 0.5, 0.3, 0.1`.

### Switching Compressor Type

The compressor paradigm is controlled by the `COMPRESSOR_TYPE` environment variable:

```bash
# Extractive compression (default)
export COMPRESSOR_TYPE=llmlingua2

# Abstractive compression
export COMPRESSOR_TYPE=abstractive
```

When `COMPRESSOR_TYPE=abstractive`, the pipeline automatically uses `facebook/bart-large-cnn` regardless of the `COMPRESSOR_MODEL` setting.

## Requirements

The project uses **Conda**. Main dependencies:

1. `torch`
2. `transformers`
3. `datasets`
4. `llmlingua`
5. `accelerate`
6. `sentencepiece`
7. `openai` (API backend only)

See [requirements.txt](requirements.txt).

## Installation

```bash
conda activate tesis
pip install -r requirements.txt
```

## Running the Pipeline

### Local pilot test

```bash
conda activate tesis
python pipeline/run_experiment.py \
  --profile pilot \
  --backend local \
  --datasets jbb \
  --rates 0.9,0.7,0.5,0.3,0.1 \
  --num-samples 10 \
  --no-prompt-guard \
  --delay 0.0
```

### Full server experiment (extractive)

```bash
conda activate tesis
export COMPRESSOR_TYPE=llmlingua2
python pipeline/run_experiment.py \
  --profile server \
  --backend api \
  --datasets jbb,advbench \
  --rates 0.9,0.7,0.5,0.3,0.1 \
  --num-samples 100 \
  --template dan
```

### Full server experiment (abstractive)

```bash
conda activate tesis
export COMPRESSOR_TYPE=abstractive
python pipeline/run_experiment.py \
  --profile server \
  --backend api \
  --datasets jbb,advbench \
  --rates 0.9,0.7,0.5,0.3,0.1 \
  --num-samples 100 \
  --template dan
```

## CLI Arguments

| Argument | Description |
|----------|-------------|
| `--profile` | Execution profile: `pilot`, `lab`, `laptop`, `server` |
| `--backend` | `local` (Transformers) or `api` (OpenAI-compatible) |
| `--datasets` | Comma-separated dataset aliases |
| `--rates` | Comma-separated compression rates |
| `--num-samples` | Samples per dataset |
| `--template` | Jailbreak template: `raw`, `dev_mode`, `aim`, `dan` |
| `--use-prompt-guard` | Enable Prompt Guard detection |
| `--no-prompt-guard` | Disable Prompt Guard |
| `--delay` | Seconds between API calls |

## Output Format

Each run produces a JSON file with the following sections:

### `experiment`
Run metadata: timestamp, LLM model, judge model, compressor, datasets, rates, sample count.

### `summary`
Global summary: `asr_sin_compresion`, `asr_con_compresion`, `delta_asr`, `mcnemar_pvalue`, average latencies, average tokens, compression ratio.

### `summary_by_dataset`
Metrics grouped by dataset.

### `summary_by_dataset_rate`
Metrics grouped by dataset and compression rate.

### `results`
Per-sample data: dataset, rate, original and compressed prompts, model responses, judge verdicts for both conditions.

## Key Metrics

### ASR (Attack Success Rate)
Percentage of attacks that succeed (judge labels response as `unsafe`).

### Delta ASR
$$\Delta ASR = ASR_{compressed} - ASR_{uncompressed}$$

If $\Delta ASR > 0$, compression made the attack more successful.

### McNemar p-value
Exact paired binomial test comparing responses on the same prompt under two conditions. Bonferroni-corrected threshold: $\alpha_{adj} = 0.0016$.

## Organized Results

Experimental results are in `results_final/`, organized by compression paradigm and model:

| Compressor | Model | Templates | Files |
|-----------|-------|-----------|-------|
| **LLMLingua-2** (Extractive) | Llama 4 Scout 109B | raw, dev_mode, aim, dan | 4 |
| | Qwen 1.5 110B | raw, dev_mode, aim, dan | 4 |
| | Gemma 2 27B | raw, dev_mode, aim, dan | 4 |
| **BART-large-cnn** (Abstractive) | Llama 4 Scout 109B | raw, dev_mode, aim, dan | 4 |
| | Qwen 1.5 110B | raw, dev_mode, aim, dan | 4 |
| | Gemma 2 27B | raw, dev_mode, aim, dan | 4 |

**Total: 24 files** (~1200 samples each, evaluated with Llama-Guard-3-8B).

Supplementary experiments (pilots, early runs) are archived in `results_archive/`.

## Current Limitations

1. Some dataset aliases may require adjusting the Hugging Face identifier.
2. The local judge is a practical approximation; for definitive results, use a stronger judge or human validation.
3. The local backend requires sufficient RAM/GPU for large models.
4. Models are served via Ollama with GGUF quantization (Q4_K_M); precision loss is a systematic constraint applied uniformly across all conditions.

## Repository Goal

This repository provides a reproducible experimental framework to answer a safety question in LLMs:

> Can prompt compression transform a token optimization technique into a security risk?

The experimental evidence demonstrates that compression is **not semantically neutral** regarding model security, and that the impact is determined by the interaction between compression method, compression level, and the target model's safety profile.
