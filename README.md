# Predicción de decisiones basada en actividad cerebral previa en dispositivos wearables
Proyecto de investigación sobre el libre albedrío: análisis de datos EEG previos a decisiones binarias mediante Deep Learning. Identificación de patrones cerebrales y modelado matemático del mejor algoritmo para predecir elecciones ante estímulos visuales.

## Tareas previas
1. Instala poetry en tu sistema:
    ```bash
    pipx install poetry
    ```

1. Instala las dependencias necesarias para el proyecto:
    ```bash 
    poetry install
    ```
1. Preprocesa los datos incluidos en ```./data/Muse EEG Subconscious Decisions Dataset```:

    ```bash
    poetry run nohup python scripts/data_preprocessor.py &
    ```
