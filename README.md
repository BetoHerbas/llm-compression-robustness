# Thesis Pipeline: Prompt Compression and Security in LLMs

## Models Used

| Role | Model | Parameters | Quantization |
|------|-------|-----------|--------------|
| **Target model (victim)** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | INT4 (nf4) |
| **Primary Judge** | `meta-llama/Llama-Guard-3-8B` | 8B | INT4 (nf4) |
| **Compressor** | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` | 178M | FP32 (CPU) |
| **Optional Detector** | `meta-llama/Prompt-Guard-86M` | 86M | FP32 (CPU) |

## Research Objective

This project implements an experimental pipeline to study how prompt compression affects the safety of Large Language Models (LLMs), focusing on prompt injection and jailbreak attacks.

The central thesis evaluates whether a prompt compression technique — **LLMLingua-2** — can alter a target model's behavior so that malicious attacks become:
1. More successful.
2. Harder to detect.
3. Cheaper in tokens and latency.

The guiding hypothesis is that **compression does not necessarily act as a defense**, and in certain scenarios it can degrade system safety by preserving malicious intent while altering the linguistic signals that help reject the attack.

## Research Question

> Does prompt compression increase the Attack Success Rate (ASR) of jailbreak or prompt injection attacks on an LLM, compared to using the original uncompressed prompt?

## Working Hypothesis

> Given a set of malicious prompts, the compressed version can produce a higher **Attack Success Rate (ASR)** than the original, especially at more aggressive compression levels.

Formally:
- $H_0$: $ASR_{compressed} \le ASR_{uncompressed}$
- $H_1$: $ASR_{compressed} > ASR_{uncompressed}$

## What This Pipeline Does

For each attack, the pipeline compares two conditions:
1. The original prompt, uncompressed.
2. The same prompt after compression with LLMLingua-2 (or BART for abstractive mode).

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
│   ├── CompressionAttack.pdf
│   └── generate_figures.py
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
│   ├── perplexity/                #   Perplexity-based compressor tests
│   ├── otros_modelos/             #   Pilot experiments (smaller models)
│   └── README.md
├── dashboard/                     # Interactive visualization dashboard
├── experiments/                   # Run scripts and configuration
│   ├── scripts/                   #   Experiment launch scripts
│   ├── setup/                     #   Dependency installation
│   ├── archive_scripts/           #   Early test scripts
│   └── logs/                      #   Execution logs
├── resultados/                    # Active pipeline output (gitignored)
└── docs/                          # Supporting documentation (Spanish)
    ├── contexto_tesis.md
    └── analisis_cientifico.md
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
1. `jbb` — JailbreakBench/JBB-Behaviors
2. `advbench` — walledai/AdvBench
3. `securitylingua`
4. `compressionattack`
5. `partprompt`

### Compression Rates

The pipeline supports rate sweeping: `0.9, 0.7, 0.5, 0.3, 0.1`.

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
  --rates 0.9,0.7,0.5,0.3 \
  --num-samples 10 \
  --no-prompt-guard \
  --delay 0.0
```

### Full lab experiment

```bash
conda activate tesis
python pipeline/run_experiment.py \
  --profile lab \
  --backend local \
  --datasets jbb,advbench \
  --rates 0.9,0.7,0.5,0.3 \
  --num-samples 100 \
  --use-prompt-guard \
  --delay 0.3
```

### Server run (Ollama)

```bash
conda activate tesis
python pipeline/run_experiment.py \
  --profile server \
  --backend api \
  --datasets jbb \
  --rates 0.9,0.7,0.5,0.3,0.1 \
  --num-samples 100
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

Each run produces a JSON file in `resultados/` with the following sections:

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
Exact paired binomial test comparing responses on the same prompt under two conditions.

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

Supplementary experiments (pilots, perplexity, early runs) are archived in `results_archive/`.

## Additional Documentation

- [`docs/contexto_tesis.md`](docs/contexto_tesis.md) — Full research context (Spanish)
- [`docs/analisis_cientifico.md`](docs/analisis_cientifico.md) — Detailed scientific analysis (Spanish)

## Current Limitations

1. Some dataset aliases may require adjusting the Hugging Face identifier.
2. The local judge is a practical approximation; for definitive results, use a stronger judge or human validation.
3. The local backend requires sufficient RAM/GPU for large models.
4. Some Llama 4 model variants are not as readily available as smaller open models.

## Repository Goal

This repository provides a reproducible experimental framework to answer a safety question in LLMs:

> Can prompt compression transform a token optimization technique into a security risk?

The value of this thesis lies in demonstrating — or refuting — this hypothesis with controlled experimental evidence.
