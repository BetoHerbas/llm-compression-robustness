# Resultados Archivados — Experimentos Adicionales

Contiene experimentos que **no forman parte directa del paper** pero se preservan
como evidencia del proceso de investigación.

## Contenido

```
results_archive/
├── resultados_server/      # Versiones tempranas de llama4:scout en servidor
│                           # (1000 muestras vs 1200 en results_final/)
├── perplexity/             # Compresor basado en Perplexity (GPT-2)
│                           # No usado en paper por límite de contexto (1024 tokens)
└── otros_modelos/          # Pruebas piloto con modelos más pequeños:
    ├── Meta-Llama-3-8B-Instruct (API Together)
    ├── Mistral-7B-Instruct-v0.1
    ├── Qwen2.5-0.5B-Instruct / 7B-Instruct
    ├── Meta-Llama-3.1-8B-Instruct GGUF (servidor)
    └── llama4:scout (1000 muestras, temprano)
```

Los resultados finales del paper están en `../results_final/`.
