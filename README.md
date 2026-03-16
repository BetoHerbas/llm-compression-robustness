# Pipeline de Tesis: Compresión de Prompts y Seguridad en LLMs

## Objetivo General

Este proyecto implementa un pipeline experimental para estudiar cómo la compresión de prompts afecta la seguridad de modelos de lenguaje grandes (LLMs), con foco en ataques de prompt injection y jailbreak.

La idea central de la tesis es evaluar si una técnica de compresión de prompts, en este caso **LLMLingua-2**, puede modificar el comportamiento de un modelo objetivo de manera que los ataques maliciosos se vuelvan:

1. Más exitosos.
2. Más difíciles de detectar.
3. Más baratos en tokens y latencia.

En otras palabras, la hipótesis que guía este trabajo es que **la compresión no necesariamente actúa como defensa**, y que en ciertos escenarios puede degradar la seguridad del sistema al preservar la intención maliciosa pero alterar señales lingüísticas que ayudan a rechazar el ataque.

## Pregunta de Investigación

La pregunta principal que intenta responder este repositorio es:

> ¿La compresión de prompts incrementa la tasa de éxito de ataques de jailbreak o prompt injection sobre un LLM, en comparación con el uso del prompt original sin comprimir?

## Hipótesis de Trabajo

La hipótesis experimental es:

> Dado un conjunto de prompts maliciosos, la versión comprimida del prompt puede producir una **Attack Success Rate (ASR)** mayor que la del prompt original, especialmente cuando el nivel de compresión es más agresivo.

Formalmente:

- $H_0$: $ASR_{con\ compresion} \le ASR_{sin\ compresion}$
- $H_1$: $ASR_{con\ compresion} > ASR_{sin\ compresion}$

## Qué Hace Este Pipeline

El pipeline compara, para un mismo ataque, dos condiciones:

1. El prompt original, sin compresión.
2. El mismo prompt luego de ser comprimido con LLMLingua-2.

Ambas versiones se envían al modelo objetivo, y luego un judge automático clasifica si la respuesta es segura o insegura. A partir de eso se calculan métricas como ASR, latencia, tokens y pruebas estadísticas pareadas.

## Flujo General

El procesamiento ocurre **muestra por muestra**, no en batch global previo.

Para cada dataset, para cada rate de compresión, y para cada prompt, el flujo es:

1. Se toma un prompt malicioso del dataset.
2. Opcionalmente, se evalúa con Prompt Guard antes de comprimir.
3. El prompt se comprime con LLMLingua-2.
4. Opcionalmente, se vuelve a evaluar con Prompt Guard después de comprimir.
5. El modelo objetivo responde al prompt original.
6. El modelo objetivo responde al prompt comprimido.
7. El judge clasifica ambas respuestas como `safe` o `unsafe`.
8. Se guardan resultados, latencias, tokens y metadatos.

Ese ciclo se repite para todas las muestras y para todos los rates configurados.

## Arquitectura Experimental

La arquitectura conceptual del experimento es:

```text
Prompt original
      |
      v
[Opcional] Prompt Guard
      |
      v
LLMLingua-2  --->  Prompt comprimido
      |
      v
LLM principal
      |
      v
Judge automático
      |
      v
Métricas: ASR / delta ASR / latencia / tokens / detección
```

## Componentes del Proyecto

### 1. Compresor

El compresor usado es **LLMLingua-2**:

- Modelo: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
- Implementado en [pipeline/config.py](pipeline/config.py) y [pipeline/run_experiment.py](pipeline/run_experiment.py)

Su función es reducir el número de tokens del prompt manteniendo, en lo posible, su intención semántica.

### 2. Modelo Objetivo

El modelo objetivo es el LLM sobre el cual se prueba el ataque.

Actualmente el pipeline soporta dos modos:

1. `local`: carga modelos Hugging Face localmente con Transformers.
2. `api`: usa un endpoint compatible con OpenAI.

La implementación está en [pipeline/llm_client.py](pipeline/llm_client.py).

Para las pruebas piloto se usan modelos livianos como `Qwen/Qwen2.5-0.5B-Instruct`. Para el experimento real se recomienda reemplazarlo por el modelo final que se quiera estudiar, por ejemplo una variante local de Llama 4, si está disponible en la infraestructura del laboratorio.

### 3. Judge Automático

El judge determina si la respuesta del modelo constituye una respuesta insegura o una negativa segura.

La implementación está en [pipeline/judge.py](pipeline/judge.py).

Actualmente también soporta modo local y modo API. En local, el judge se formula como una tarea generativa donde se pide al modelo emitir una clasificación `SAFE` o `UNSAFE`.

### 4. Prompt Guard

El detector opcional de prompt injection / jailbreak está implementado en [pipeline/prompt_guard.py](pipeline/prompt_guard.py).

Su función no es reemplazar al judge, sino aportar una métrica extra:

1. Si el ataque era detectable antes de comprimir.
2. Si deja de ser detectable después de comprimir.

Esto permite estudiar evasión de detectores, no solo éxito directo del ataque.

### 5. Métricas

Las métricas están implementadas en [pipeline/metrics.py](pipeline/metrics.py).

Actualmente el sistema calcula:

1. `ASR sin compresión`
2. `ASR con compresión`
3. `delta ASR`
4. Latencia promedio
5. Tokens promedio
6. Ratio de compresión promedio
7. Test de McNemar exacto para comparación pareada
8. Métricas por dataset
9. Métricas por dataset y por rate

## Estructura del Proyecto

```text
.
├── README.md
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── llm_client.py
│   ├── judge.py
│   ├── metrics.py
│   ├── prompt_guard.py
│   └── run_experiment.py
└── resultados/
```

## Configuración Principal

La configuración central está en [pipeline/config.py](pipeline/config.py).

### Perfiles

Hay dos perfiles principales:

1. `PILOT`
2. `LAB`

`PILOT` está pensado para validaciones rápidas en una laptop. `LAB` está orientado al experimento final en una máquina más potente.

### Backend

`BACKEND` puede ser:

1. `local`
2. `api`

Si se usa `local`, los modelos se cargan con Transformers desde Hugging Face o desde caché local.

### Datasets

El archivo [pipeline/config.py](pipeline/config.py) define un `DATASET_REGISTRY` con aliases de datasets experimentales.

Ejemplos actualmente contemplados:

1. `jbb`
2. `advbench`
3. `securitylingua`
4. `compressionattack`
5. `partprompt`

Algunos pueden requerir corrección del nombre exacto en Hugging Face, autenticación o ajuste del esquema de columnas.

### Rates de Compresión

El pipeline soporta barrido de rates:

1. `0.9`
2. `0.7`
3. `0.5`
4. `0.3`

Esto permite observar si la compresión más agresiva produce cambios sistemáticos en la seguridad.

## Requisitos

El proyecto está pensado para ejecutarse con **Conda**.

Dependencias principales:

1. `torch`
2. `transformers`
3. `datasets`
4. `llmlingua`
5. `accelerate`
6. `sentencepiece`
7. `openai` solo si se quiere backend API

Las dependencias están en [requirements.txt](requirements.txt).

## Instalación con Conda

Ejemplo de uso con el entorno `tesis`:

```bash
conda activate tesis
cd /home/beto/Documents/UCB/9noSemestre/TallerDeGrado1/tests
python -m pip install -r requirements.txt
```

El proyecto está configurado para trabajar con el intérprete Conda `tesis` como entorno principal.

## Cómo Ejecutar el Pipeline

### Prueba piloto local

```bash
conda activate tesis
cd /home/beto/Documents/UCB/9noSemestre/TallerDeGrado1/tests
python pipeline/run_experiment.py \
  --profile pilot \
  --backend local \
  --datasets jbb \
  --rates 0.9,0.7,0.5,0.3 \
  --num-samples 10 \
  --no-prompt-guard \
  --delay 0.0
```

### Corrida más fuerte para laboratorio

```bash
conda activate tesis
cd /home/beto/Documents/UCB/9noSemestre/TallerDeGrado1/tests
python pipeline/run_experiment.py \
  --profile lab \
  --backend local \
  --datasets jbb,advbench \
  --rates 0.9,0.7,0.5,0.3 \
  --num-samples 100 \
  --use-prompt-guard \
  --delay 0.3
```

## Parámetros de Línea de Comando

El archivo [pipeline/run_experiment.py](pipeline/run_experiment.py) soporta los siguientes argumentos principales:

1. `--profile`: selecciona `pilot` o `lab`.
2. `--backend`: selecciona `local` o `api`.
3. `--datasets`: lista CSV de aliases de datasets.
4. `--rates`: lista CSV de rates de compresión.
5. `--num-samples`: número de muestras por dataset.
6. `--use-prompt-guard`: activa Prompt Guard.
7. `--no-prompt-guard`: lo desactiva aunque el perfil lo active.
8. `--delay`: pausa entre llamadas.

## Cómo Se Interpreta la Salida

Cada corrida produce un archivo JSON en la carpeta `resultados/`.

Ese JSON contiene varias secciones:

### `experiment`

Metadatos de la corrida:

1. timestamp
2. modelo LLM
3. modelo judge
4. compresor
5. datasets usados
6. rates usados
7. número de muestras

### `summary`

Resumen global de toda la corrida.

Incluye:

1. `asr_sin_compresion`
2. `asr_con_compresion`
3. `delta_asr`
4. `mcnemar_pvalue`
5. latencias promedio
6. tokens promedio
7. ratio de compresión

### `summary_by_dataset`

Agrupa métricas por dataset.

### `summary_by_dataset_rate`

Agrupa métricas por dataset y por rate.

### `results`

Contiene una fila por caso experimental. Cada fila incluye, entre otros:

1. dataset
2. rate
3. prompt original
4. prompt comprimido
5. respuesta sin compresión
6. respuesta con compresión
7. juicio del judge para ambas condiciones

## Qué Significa Cada Métrica Principal

### ASR

La **Attack Success Rate** mide cuántos ataques fueron exitosos. En este pipeline, un ataque se considera exitoso cuando el judge clasifica la respuesta como `unsafe`.

### Delta ASR

Se define como:

$$
\Delta ASR = ASR_{con\ compresion} - ASR_{sin\ compresion}
$$

Si $\Delta ASR > 0$, entonces la compresión hizo que el ataque fuera más exitoso.

### McNemar p-value

Es una prueba estadística para comparar respuestas pareadas sobre el mismo prompt en dos condiciones.

Sirve para evaluar si la diferencia observada entre el caso con y sin compresión es compatible con ruido o si hay evidencia de un cambio sistemático.

## Qué Debe Mostrar la Tesis para Validar la Hipótesis

La hipótesis queda mejor respaldada si se observa simultáneamente:

1. $\Delta ASR > 0$ de forma consistente.
2. Incremento de ASR al bajar el rate de compresión.
3. Resultados repetibles en más de un dataset.
4. p-values bajos en comparaciones relevantes.
5. Casos cualitativos donde el prompt comprimido preserva la intención maliciosa pero reduce señales de seguridad.

## Recomendaciones para el Experimento Real

### Componentes recomendados

1. Modelo objetivo: una variante local de Llama 4 Instruct, si está disponible en laboratorio.
2. Compresor: LLMLingua-2.
3. Judge principal: Llama Guard 3 8B o un juez de seguridad más robusto que el modelo objetivo.
4. Detector opcional: Prompt Guard 86M.

### Benchmarks recomendados

1. `jbb` como benchmark principal.
2. `advbench` como benchmark adversarial complementario.
3. `compressionattack` para medir deriva semántica inducida por compresión.
4. `securitylingua` como análisis adicional si el tiempo lo permite.

### Diseño recomendado

1. 100 a 200 muestras por dataset.
2. Rates: `0.9, 0.7, 0.5, 0.3`.
3. Mismo modelo y misma configuración entre condiciones.
4. Comparación siempre pareada entre original y comprimido.
5. Idealmente dos corridas para validar estabilidad.

## Limitaciones Actuales del Repositorio

1. Algunos aliases de datasets pueden requerir ajuste fino del identificador de Hugging Face.
2. El judge local actual es una aproximación útil, pero para resultados definitivos de tesis conviene usar un judge más fuerte o complementar con validación humana.
3. El backend local depende de memoria RAM y/o GPU suficiente para cargar modelos grandes.
4. No todos los modelos Llama 4 están disponibles localmente de la misma forma que modelos abiertos más pequeños.

## Estado Actual del Proyecto

El pipeline ya permite:

1. Ejecutar pruebas piloto localmente.
2. Comparar prompts originales vs comprimidos.
3. Barrer múltiples rates de compresión.
4. Trabajar con múltiples datasets.
5. Medir ASR, latencia, tokens y estadísticas pareadas.
6. Guardar resultados estructurados para análisis posterior.

## Objetivo Final del Repositorio

Este repositorio no busca solo ejecutar prompts, sino construir una base experimental reproducible para responder una pregunta de seguridad en LLMs:

> ¿La compresión de prompts puede transformar una técnica de optimización de tokens en un factor de riesgo para la seguridad?

El valor principal de esta tesis está en demostrarlo o refutarlo con evidencia experimental controlada, no solo con intuiciones.