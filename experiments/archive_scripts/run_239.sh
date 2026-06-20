#!/bin/bash
cd /home/ubuntu/tesis_experiments
source venv/bin/activate

echo "Waiting for qwen:110b to be downloaded..."
until ollama list | grep -q "qwen:110b"; do sleep 30; done
echo "Model ready! Starting experiments."

export LLM_MODEL="qwen:110b"
export NUM_SAMPLES=100

# Qwen + Perplexity
export COMPRESSOR_TYPE="perplexity"
for TEMPLATE in raw dev_mode aim dan; do
    export TEMPLATE
    echo "Running Qwen with Perplexity - Template: $TEMPLATE"
    python3 pipeline/run_experiment.py --profile server --datasets jbb,advbench --rates 0.9,0.7,0.5,0.3,0.1 --delay 0.0
done

# Qwen + LLMLingua-2
export COMPRESSOR_TYPE="llmlingua2"
for TEMPLATE in raw dev_mode aim dan; do
    export TEMPLATE
    echo "Running Qwen with LLMLingua-2 - Template: $TEMPLATE"
    python3 pipeline/run_experiment.py --profile server --datasets jbb,advbench --rates 0.9,0.7,0.5,0.3,0.1 --delay 0.0
done

echo "ALL DONE ON .239" > all_done.flag
