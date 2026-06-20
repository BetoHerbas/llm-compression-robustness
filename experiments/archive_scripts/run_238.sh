#!/bin/bash
cd /home/ubuntu/tesis_experiments
source venv/bin/activate

echo "Waiting for gemma2:27b to be downloaded..."
until ollama list | grep -q "gemma2:27b"; do sleep 30; done
echo "Model ready! Starting experiments."

# Llama + Perplexity
export COMPRESSOR_TYPE="perplexity"
export LLM_MODEL="llama4:scout"
export NUM_SAMPLES=100
for TEMPLATE in raw dev_mode aim dan; do
    export TEMPLATE
    echo "Running Llama with Perplexity - Template: $TEMPLATE"
    python3 pipeline/run_experiment.py --profile server --datasets jbb,advbench --rates 0.9,0.7,0.5,0.3,0.1 --delay 0.0
done

# Gemma + LLMLingua-2
export COMPRESSOR_TYPE="llmlingua2"
export LLM_MODEL="gemma2:27b"
for TEMPLATE in raw dev_mode aim dan; do
    export TEMPLATE
    echo "Running Gemma with LLMLingua-2 - Template: $TEMPLATE"
    python3 pipeline/run_experiment.py --profile server --datasets jbb,advbench --rates 0.9,0.7,0.5,0.3,0.1 --delay 0.0
done

# Gemma + Perplexity (only dan)
export COMPRESSOR_TYPE="perplexity"
export TEMPLATE="dan"
echo "Running Gemma with Perplexity - Template: $TEMPLATE"
python3 pipeline/run_experiment.py --profile server --datasets jbb,advbench --rates 0.9,0.7,0.5,0.3,0.1 --delay 0.0

echo "ALL DONE ON .238" > all_done.flag
