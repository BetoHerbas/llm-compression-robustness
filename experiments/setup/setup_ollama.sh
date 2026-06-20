#!/bin/bash

PASS="Cuchitril2020."

echo ">>> Instalando Ollama..."
echo "$PASS" | sudo -S bash -c 'curl -fsSL https://ollama.com/install.sh | sh'

echo ">>> Verificando servicio Ollama..."
sleep 5
systemctl is-active --quiet ollama || (echo "Ollama no inició correctamente" && exit 1)

echo ">>> Iniciando descarga de Llama 4 Scout (109B)..."
nohup ollama pull llama4:scout > ollama_pull_llama4.log 2>&1 &

echo ">>> Iniciando descarga de Llama Guard 3 (8b)..."
nohup ollama pull llama-guard3:8b > ollama_pull_guard.log 2>&1 &

echo ">>> Estado actual de Ollama:"
ollama list

echo ">>> Proceso de descarga iniciado en segundo plano. Monitorear con ollama list o los logs localmente."
