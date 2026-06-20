#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
VENV_PATH="/home/ubuntu/tesis_experiments/venv"

echo ">>> Activando VENV..."
source $VENV_PATH/bin/activate

echo ">>> Instalando dependencias (Force)..."
# Usamos pip directamente desde el binario para evitar confusiones
$VENV_PATH/bin/python3 -m pip install --upgrade pip
$VENV_PATH/bin/python3 -m pip install openai requests transformers datasets tqdm

echo ">>> Instalando Torch y LLMLingua (Paso crítico)..."
$VENV_PATH/bin/python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$VENV_PATH/bin/python3 -m pip install llmlingua

echo ">>> Verificando instalación..."
$VENV_PATH/bin/python3 -c "import torch; import transformers; import llmlingua; print('Depedencias OK')"

echo ">>> Lista de paquetes final:"
$VENV_PATH/bin/pip list
