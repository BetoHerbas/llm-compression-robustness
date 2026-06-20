#!/bin/bash
cd /home/ubuntu/tesis_experiments
source venv/bin/activate

# Llama 4 Scout + Perplexity
export COMPRESSOR_TYPE="perplexity"
export COMPRESSOR_MODEL="gpt2"
export LLM_MODEL="llama4:scout"
export NUM_SAMPLES=100

echo "Starting Llama 4 Scout with Perplexity (gpt2 compressor model) on all templates..."

for TEMPLATE in raw dev_mode aim dan; do
    export TEMPLATE
    echo "Running Llama with Perplexity - Template: $TEMPLATE"
    python3 pipeline/run_experiment.py --profile server --datasets jbb,advbench --rates 0.9,0.7,0.5,0.3,0.1 --delay 0.0
done

echo "ALL PERPLEXITY EXPERIMENTS DONE ON .238" > perplexity_done.flag
