# Predicción de decisiones basada en EEG con dispositivos wearables

Proyecto de investigación que explora el libre albedrío humano a través del análisis de datos EEG captados por dispositivos wearables. Se investiga, mediante técnicas de Deep Learning (redes neuronales), si es posible predecir decisiones binarias (P/Q) tras presentar a los individuos un estímulo visual. Se pretende, además, modelar matemáticamente el algoritmo que resulte más eficaz para la predicción de decisiones.

---
## Índice
1. [Estado actual del proyecto](#estado-actual-del-proyecto)
   1. [Estructura del repositorio](#estructura-del-repositorio)
   1. [Dependencias e instalación](#dependencias-e-instalación)
   1. [Uso de los scripts](#uso-de-los-scripts)
1. [Flujo de trabajo del proyecto](#flujo-de-trabajo-del-proyecto)
1. [Próximos pasos](#próximos-pasos)
   1. [Métricas a estudiar](#métricas-a-estudiar)

---

## Estado actual del proyecto

### Estructura del repositorio
El repositorio está organizado de la siguiente manera:

<!-- TODO: Actualizar conforme se avance -->
```text
.
├── data/
│   └── Muse EEG Subconscious Decisions Dataset/           <-- Dataset original
│   │   ├── Local              <-- Decisiones y tiempos de respuestas (resultsX.csv)
│   │   └── Muse               <-- Datos EEG y de sensores registrados (museDataX.csv)
│   ├── processed/             <-- Directorio de salida de los datos preprocesados
│   ├── splits.csv             <-- Asignación de respuestas a train/test y fold
│   └── subjects.csv           <-- Recuento de respuestas por sujeto y ventana
├── prediccion_de_decisiones/  <-- Carpeta con los modelos y el entrenamiento
├── scripts/                   <-- Scripts extras del proyecto
│   ├── data_preprocessor.py   <-- Preprocesado, normalización y segmentación en ventanas
│   └── sets_creator.py        <-- Creación de splits (train, val, test) y folds
├── tests/                     <-- Scripts y ficheros de pruebas
├── config.ini                 <-- Configuraciones generales (rutas, ventana, etc.)
├── poetry.lock                <-- Archivo de bloqueo de dependencias
├── pyproject.toml             <-- Configuración del proyecto
├── README.md                  <-- Este documento
└── .gitignore                 <-- Archivos ignorados por git
```

### Dependencias e instalación

Este proyecto usa **Poetry** para manejar dependencias. Para instalarlo y luego configurar el entorno:

1. Instala ```poetry``` en tu sistema (si no lo tienes):
    ```bash
    pipx install poetry
    ```

1. En la raíz del proyecto, instala las dependencias necesarias para este proyecto:
    ```bash
    poetry install
    ```

1. Preprocesa los datos incluidos en ```./data/Muse EEG Subconscious Decisions Dataset```:
    ```bash
    poetry run nohup python scripts/data_preprocessor.py &
    ```

<!-- TODO: Continuar añadiendo los pasos para replicar el trabajo -->
1. PENDIENTE

<!-- TODO: Actualizar si se utiliza otra librería de Deep Learning diferente a PyTorch (como TensorFlow, etc.) -->
Se requiere, al menos:
- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- PyTorch

### Uso de los scripts

## Flujo de trabajo del proyecto
1. Descargar/Verificar dataset en ```data/Muse EEG Subconscious Decisions Dataset```.
1. Preprocesar (usando ```data_preprocessor.py```):
   - Normalización de señales.
   - Segmentación en ventanas.
   - Asignación de sujetos a train/test y a folds de validación.
1. Entrenar modelos (futuro):
   - Utilizar PyTorch para crear CNNs/LSTMs.
   - Cargar datos del set train, validar con folds; finalmente usar test para resultados finales.
1. Analizar resultados (futuro):
   - Calcular Accuracy, F1-score, G-Mean, etc.
   - Comparar rendimientos entre diferentes longitudes de ventana.

## Próximos pasos

1. Implementar entrenamientos: Crear un script de entrenamiento (en la carpeta ```prediccion_de_decisiones```) que use PyTorch.

1. Realizar pruebas con distintas ventanas (100ms, 250ms, etc.) y recopilar métricas.

1. Optimizar hiperparámetros (número de capas, learning rate, etc.).

1. Documentar resultados en el TFG, con tablas, gráficas y conclusiones finales.

### Métricas a estudiar

- **G-Mean**: Útil con datos desbalanceados.
- **Accuracy**: Proporción de aciertos globales.
- **F1-Score** (**macro** y **micro**): Evalúa la combinación de precisión y exhaustividad.
- **Sensibilidad** (**Recall**) y **Especificidad**: Indican qué tan bien distingue clases.
- **Precisión**: Nivel de acierto en lo que el modelo predice como positivo.