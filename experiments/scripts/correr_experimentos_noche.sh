#!/bin/bash
# Script extendido para correr TODOS los experimentos cruzados
# Permite probar la eficacia de los Jailbreaks Comprimidos a través de distintas familias de modelos.

# Aseguramos la API Key de Together AI
export TOGETHER_API_KEY="tgp_v1_3xio8O4iKaZ1V4VFTMpd2UUeapdIwtSMon5N1w_iMQo"

# Configurar conda para activar el entorno "tesis" automáticamente
eval "$(conda shell.bash hook)"
conda activate tesis

echo "========================================================="
echo " 🚀 INICIANDO MEGA BATCH DE EXPERIMENTOS DE TESIS (OVERNIGHT)"
echo "    Dataset: JailbreakBench (JBB - 100 muestras reales)"
echo "========================================================="
echo ""

# Definición de la matriz de variables
MODELOS=(
  "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
  "mistralai/Mistral-7B-Instruct-v0.1"
  "Qwen/Qwen2.5-7B-Instruct-Turbo"
)

TEMPLATES=(
  "raw"
  "dev_mode"
  "aim"
  "dan"
)

# Iteración anidada: Por cada modelo, probar cada plantilla.
for MODELO in "${MODELOS[@]}"; do
    echo "========================================================="
    echo " 🧠 Evaluando Familia de Modelo: $MODELO"
    echo "========================================================="
    
    # Exportar el modelo como variable de entorno para que config.py la intercepte
    export LLM_MODEL="$MODELO"
    
    # Derivar un prefijo seguro para el archivo de logs
    LOG_PREFIX=$(echo "$MODELO" | cut -d'/' -f2 | awk '{print tolower($0)}')
    
    for TEMPLATE in "${TEMPLATES[@]}"; do
        echo "⏳ Ejecutando Template: [ $TEMPLATE ] sobre [ $LOG_PREFIX ]..."
        
        LOG_FILE="run_${LOG_PREFIX}_${TEMPLATE}.log"
        
        # Ejecución silenciada
        python pipeline/run_experiment.py --profile laptop --template $TEMPLATE --num-samples 100 --rates 0.9,0.6,0.4,0.2 > "$LOG_FILE" 2>&1
        
        echo " ✔ Finalizado exitosamente. Log en $LOG_FILE"
        echo "---------------------------------------------------------"
    done
done

echo ""
echo "========================================================="
echo " 🎉 TODOS LOS EXPERIMENTOS CRUZADOS FINALIZADOS EXITOSAMENTE"
echo " 📁 Revisa la carpeta 'resultados/' para encontrar decenas de archivos JSON."
echo "========================================================="
