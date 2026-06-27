# FIS205 - Tarea 2

Repositorio para la Tarea 2 de Física Computacional.

## Contenido

- Problema 1: problema inverso para un oscilador amortiguado.
- Problema 2: simulación del pico de Bragg para protones en agua.

## Estructura

- `src/`: scripts de Python.
- `outputs/figures/`: figuras generadas.
- `data/`: datos generados localmente, no incluidos en GitHub.

## Ejecución

```bash
python3 src/p1_generate_data.py
python3 src/p1_train_models.py
python3 src/p1_noise_study.py

python3 src/p2_bethe_csda.py
python3 src/p2_bragg_determinista.py
python3 src/p2_bragg_straggling.py
