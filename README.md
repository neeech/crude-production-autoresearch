# Autoresearch: petróleo no convencional

Pronóstico de producción mensual de petróleo por pozo, en m³, con experimentos
de features y modelos. Se predice al inicio del mes t usando registros de t−2
o anteriores cargados hasta ese instante. El objetivo es minimizar MAE de
validación. RMSE se registra como diagnóstico.

## Preparar y comprobar

Requiere Python >=3.13, uv y Git configurado. Desde la raíz:

```powershell
uv sync
uv run python -m crude_production_autoresearch.preprocess --download
uv run pytest
uv run python -m crude_production_autoresearch.train
```

La suite de tests usa datos sintéticos y no requiere descargar el dataset ni
configurar claves de API. Para ver cada caso, usar `uv run pytest -v`.

La descarga oficial de producción no convencional se guarda en
`data/raw/production_unconventional.csv`. `--download` descarga sólo si ese archivo
no existe; sin esa opción se usa el snapshot local. Para cambiar de snapshot,
conservar una copia y su hash antes de reemplazarlo. La ingesta registra URL,
SHA-256, configuración y tamaños efectivos en `data/processed_oil/manifest.json`.
El CSV y las particiones nuevas se ignoran en Git. Los datos anteriores de compra
no son inputs de la tarea nueva.

Fuente: [Secretaría de Energía — Capítulo IV](http://datos.energia.gob.ar/dataset/produccion-de-petroleo-y-gas-por-pozo).
La URL directa está fijada en `preprocess.py` y el manifiesto.

## Contrato de predicción

Cada fila es un pozo-mes con target de petróleo observado no negativo. Para entrar
al universo, su registro t−2 debe tener petróleo positivo, tipo petrolífero,
cuenca Neuquina y carga registrada antes o al inicio de t. Los ceros futuros se
conservan. Un mes ausente no se convierte en cero. Esto evalúa continuidad de
pozos existentes; no productividad inicial de pozos nuevos ni causalidad de fallas.

El input incluye calendario del mes objetivo, petróleo de t−2, contexto del registro
t−2 y una lista de observaciones entre t−2 y t−13. Cada registro histórico contiene
petróleo, gas, agua, tiempo efectivo, estado, extracción y distancia en meses.
La disponibilidad por fecha se aplica a cada registro. No se exponen resultados,
estado ni tiempo efectivo del mes objetivo, IDs ni fechas de carga al estimador.

Train: 2015–2022; validación: 2023–2024; test reservado: desde 2025.
Se purgan de train/validación etiquetas registradas después de comenzar la siguiente
partición. La historia se actualiza por origen, pero el modelo se ajusta sólo en
train. Los rangos efectivos dependen del snapshot, sin fabricar meses faltantes.

La normalización y los límites se explican en [program.md](program.md). La fuente
es revisada; la fecha de carga no certifica publicación pública ni reconstruye
versiones históricas. Algunos atributos pueden provenir del padrón actual.
Los resultados tienen alcance retrospectivo, no una garantía operativa.

## Archivos editables y protegidos

| Archivo | Responsabilidad | Editable por el agente |
|---|---|---|
| `features.py` | Transformaciones por fila e historia a escalares | Sí |
| `modeling.py` | Pipeline, preprocesamiento aprendido y regresor | Sí |
| `preprocess.py`, `history.py` | Normalización, universo, target, acceso al pasado y splits | No |
| `train.py`, `evaluate.py` | Entrenamiento, baselines, MAE y RMSE | No |
| `research.py` | Propuestas, commits, comparación y reanudación | No |
| `program.md`, configs y tests | Contrato e instrucciones | No |

El paquete `crude_production_autoresearch` vive en `src/` y el comando de entrada
es `autoresearch`; la tarea activa es regresión.
`build_features(frame)` preserva filas e índice; `build_model(columns, config)`
devuelve un estimador con `fit` y `predict`. La selección de columnas ocurre
después de generar features. Todo aprendizaje se ajusta sólo en train.

## Ejecutar el loop

Primero revisar y commitear los cambios: el loop exige un checkout limpio.
No hace commits automáticamente en el checkout original. Luego:

```powershell
uv run autoresearch --experiments 0
uv run autoresearch --experiments 10 --timeout 180
```

El segundo comando requiere `GROQ_API_KEY`; el proveedor predeterminado es Groq.
Para Gemini, definir `GEMINI_API_KEY` y seleccionar proveedor y modelo:

```powershell
uv run autoresearch --provider gemini --model TU_MODELO --rpm 10 --experiments 10 --timeout 600
```

`--rpm` espacia solicitudes e incluye reintentos. Las claves no se pasan al proceso
de entrenamiento. Para propuestas guardadas, sin llamadas a una API:

```powershell
uv run autoresearch --experiments 2 --proposals propuestas.json
```

El archivo contiene una lista de objetos `hypothesis`/`files` según `program.md`.
El contexto del agente incluye esquema, disponibilidad, limitaciones, splits de
train/validación, mejor código y resultados de todos los intentos. Se comparan
mediana de train, persistencia disponible y candidato inicial Ridge; después se
conservan mejoras estrictas del MAE del candidato. Un candidato puede mejorar
Ridge sin superar persistencia: ambas comparaciones deben leerse en el historial.

## Resultados y reanudación

Cada investigación crea rama y worktree aislados bajo
`experiments/autoresearch/<run_id>/`. Congela sólo train/validación y su manifiesto;
el test no se copia. Registra hashes de archivos protegidos, propuestas, commits,
scores, errores, duración y estado de las ejecuciones. `summary.json` usa
`best_mae`; `history.jsonl` contiene MAE/RMSE y la referencia de persistencia.
El candidato descartado vuelve al mejor commit. Fallos, timeouts y empates no
se conservan. Los controles de integridad no son un sandbox contra código hostil.

```powershell
uv run autoresearch --resume <run_id> --experiments 10 --provider gemini --model TU_MODELO --rpm 10 --timeout 600
```

Se requieren la carpeta completa, hashes y protocolo intactos. La reanudación
mantiene mejor commit e historial sin repetir baselines. Dos ejecuciones de la
misma investigación no pueden correr simultáneamente. Ctrl+C permite recuperar
el intento interrumpido. Las opciones de proveedor/modelo se indican de nuevo.

Las investigaciones antiguas de clasificación no se pueden reanudar con el
protocolo de MAE: iniciar una investigación nueva. No se borran sus artefactos
ni se comparan scores entre tareas. Para incorporar un resultado, revisar los
commits aceptados antes de hacer cherry-pick.

El test debe permanecer fuera de selección de features, modelos y presupuesto.
Los artefactos de `train.py` son evaluaciones de validación; no se ejecuta test
automáticamente al finalizar el loop.

## Logs de progreso

La consola muestra hora y zona horaria, número de ensayo y etapa: solicitud al
proveedor, reintento, espera por RPM, validación de propuesta, commit,
entrenamiento/evaluación y decisión. Mientras una llamada a la API o un proceso
de entrenamiento sigue activo, aparece una señal de progreso cada 15 segundos.
Esa señal indica actividad pendiente; no es un porcentaje de entrenamiento.

Los eventos de consola se guardan también en
`experiments/autoresearch/<run_id>.log` y se anexan al reanudar. El detalle del
subproceso está en `<run_id>/run_NNN/training.log`, con salida sin buffer y fases
de carga de datos, `fit`, predicción y escritura de artefactos. Para seguirlo:

```powershell
Get-Content experiments/autoresearch/<run_id>/run_NNN/training.log -Tail 30 -Wait
```

`--timeout` limita entrenamiento; cada llamada HTTP tiene un timeout de red de
180 segundos y hasta tres intentos. Con `--rpm 5` hay al menos 12 segundos entre
inicios de llamadas, incluidos reintentos. Una propuesta que agota reintentos
consume un ensayo fallido y no llega a entrenar candidato.

Estos cambios de observabilidad afectan archivos protegidos: iniciar una
investigación nueva después de commitearlos. Las investigaciones creadas con
el código anterior mantienen su protocolo anterior y no reciben estos logs
retroactivamente.
