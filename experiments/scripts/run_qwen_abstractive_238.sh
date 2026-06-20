#!/bin/bash
# ============================================================
# Qwen:110b + Abstractive  — Servidor .238
# Detiene el minero y lanza los 4 templates en background
# ============================================================

cd /home/ubuntu/tesis_experiments
source venv/bin/activate

# ── 1. Detener el minero ─────────────────────────────────────
echo "[$(date)] Deteniendo minero..."
pkill -f xmrig && echo "  xmrig terminado." || echo "  xmrig no estaba corriendo."
pkill -f xmr     && true   # por si tiene otro nombre
sleep 2

# ── 2. Verificar que qwen:110b esté disponible ───────────────
echo "[$(date)] Verificando qwen:110b en Ollama..."
until ollama list | grep -q "qwen:110b"; do
    echo "  Modelo no listo, esperando 30s..."
    sleep 30
done
echo "[$(date)] Modelo listo."

# ── 3. Configurar entorno ────────────────────────────────────
export LLM_MODEL="qwen:110b"
export COMPRESSOR_TYPE="abstractive"
export NUM_SAMPLES=100
export HF_TOKEN="${HF_TOKEN}"

RATES="0.9,0.7,0.5,0.3,0.1"
DATASETS="jbb,advbench"

# ── 4. Correr los 4 templates secuencialmente ────────────────
for TEMPLATE in raw aim dan dev_mode; do
    export TEMPLATE
    LOG="run_qwen_abstractive_${TEMPLATE}.log"
    echo "[$(date)] Iniciando qwen:110b + abstractive + ${TEMPLATE}..."
    python3 pipeline/run_experiment.py \
        --profile server \
        --datasets "$DATASETS" \
        --rates "$RATES" \
        --delay 0.0 \
        > "$LOG" 2>&1
    echo "[$(date)] Terminado: ${TEMPLATE} (log: $LOG)"
done

echo "[$(date)] === TODOS LOS TEMPLATES FINALIZADOS ===" > qwen_abstractive_done.flag
echo "[$(date)] DONE"
