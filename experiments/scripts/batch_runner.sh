#!/bin/bash
# Script para correr el experimento completo en el servidor
# Usando gemma2:27b, qwen:110b y Compresor Perplexity (gpt2)

# Entrar al directorio del proyecto (ajustar si es necesario)
cd /home/ubuntu/tesis_experiments/

# Autenticación con Hugging Face para datasets protegidos (AdvBench)
export HF_TOKEN="${HF_TOKEN}"

# Cargar entorno virtual
source venv/bin/activate

# Datasets a evaluar
DATASETS="jbb,advbench"
# Tasas de compresión
RATES="0.9,0.7,0.5,0.3,0.1"
# Muestras por dataset
SAMPLES=100

# Configurar el compresor alternativo (Perplexity-based)
export COMPRESSOR_MODEL="gpt2"
export COMPRESSOR_TYPE="perplexity"

MODELOS=(
  "gemma2:27b"
  "qwen:110b"
)

TEMPLATES=(
  "raw"
  "dev_mode"
  "aim"
  "dan"
)

echo "🚀 Asegurando modelos de Ollama (esto puede tardar la primera vez)..."
ollama pull llama-guard3
for MODELO in "${MODELOS[@]}"; do
    ollama pull "$MODELO"
done

echo "🚀 Iniciando Mega-Batch en Servidor..."
echo "Datasets: $DATASETS"
echo "Muestras: $SAMPLES"
echo "Rates:    $RATES"
echo "Compresor:$COMPRESSOR_MODEL ($COMPRESSOR_TYPE)"

cat << 'EOF' > run_all.sh
#!/bin/bash
source venv/bin/activate
for MODELO in "gemma2:27b" "qwen:110b"; do
    export LLM_MODEL="$MODELO"
    for TEMPLATE in "raw" "dev_mode" "aim" "dan"; do
        LOG_PREFIX=$(echo "$MODELO" | tr ':' '_')
        LOG_FILE="run_${LOG_PREFIX}_${TEMPLATE}.log"
        echo "⏳ [$(date)] Iniciando $MODELO con $TEMPLATE..."
        python3 pipeline/run_experiment.py --profile server --template "$TEMPLATE" --num-samples 100 --datasets "jbb,advbench" --rates "0.9,0.7,0.5,0.3,0.1" --delay 0.0 > "$LOG_FILE" 2>&1
    done
done
echo "🎉 [$(date)] TODOS LOS BATCHES FINALIZADOS"
EOF

chmod +x run_all.sh

nohup ./run_all.sh > master_experiment_log.txt 2>&1 &

PID=$!
echo "✅ Mega Batch lanzado en segundo plano (PID: $PID)"
echo "Monitorea el progreso maestro con: tail -f master_experiment_log.txt"
echo "O logs específicos de cada prueba (ej: tail -f run_gemma2_27b_raw.log)"
echo $PID > last_pid.txt
