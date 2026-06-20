#!/bin/bash
# Instalador manual de Ollama simplificado

set -e

echo ">>> Descargando Ollama..."
curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst

echo ">>> Extrayendo en /usr/local..."
mkdir -p /usr/local
tar -C /usr/local --zstd -xf /tmp/ollama.tar.zst

echo ">>> Configurando usuario y servicio..."
id -u ollama &>/dev/null || useradd -r -s /bin/false -m -d /usr/share/ollama ollama

cat <<EOF > /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=default.target
EOF

systemctl daemon-reload
systemctl enable ollama
systemctl start ollama

echo ">>> Ollama instalado y corriendo."
