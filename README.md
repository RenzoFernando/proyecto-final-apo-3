# Sistema de Clasificación Automática de Calidad y Tamaño de Frutas

## Integrantes

- Luna Catalina Martínez Vásquez
- Renzo Fernando Mosquera Daza
- Valentina Tobar Gómez

## Descripción

Este proyecto desarrolla un sistema de anotación de video e inteligencia artificial para la clasificación automática de frutas a partir de imágenes.

La solución permite estimar dos características principales:

- **Calidad:** Buena, Regular o Mala.
- **Tamaño:** Pequeño, Mediano o Grande.

Para ello se implementó un flujo completo basado en la metodología CRISP-DM que incluye adquisición de datos, análisis exploratorio, procesamiento de imágenes, construcción de variables derivadas, entrenamiento de modelos, evaluación comparativa y despliegue mediante una interfaz web.

Se evaluaron modelos de aprendizaje automático tradicional (**SVM** y **Random Forest**) y un modelo de aprendizaje profundo basado en Redes Neuronales Convolucionales (**CNN**).

## Datos utilizados

Se utilizaron dos fuentes principales de información:

- El conjunto de datos público **Fruit Quality Classification (Kaggle)**.
    -   Las instrucciones para obtener y reconstruir este conjunto de datos se encuentran en: *data/raw/kaggle/info.md*  
- Imágenes recolectadas y etiquetadas por los integrantes del proyecto.
    - Las instrucciones para acceder a estas imágenes se encuentran en: *data/raw/propias/info.md*

## Tecnologías utilizadas

- Python
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib
- HTML / CSS / JavaScript

## Metodología

El desarrollo siguió una adaptación de la metodología CRISP-DM:

1. Comprensión del problema.
2. Comprensión de los datos.
3. Preparación de los datos.
4. Modelado.
5. Evaluación.
6. Despliegue.

Entre las tareas realizadas destacan:

* Detección y eliminación de imágenes duplicadas.
* Construcción automática de la variable de tamaño mediante segmentación.
* Normalización y estandarización de imágenes.
* Búsqueda de hiperparámetros mediante GridSearchCV.
* Entrenamiento y comparación de múltiples modelos.
* Desarrollo de una aplicación web para inferencia.

## Estructura del proyecto

```text
.
├── app/
│ ├── static/
│ ├── index.html
│ └── main.py
|
├── data/
│ ├── annotations/
│ └── raw/
│   ├── kaggle/
│   └── propias/
|
├── docs/
│ ├── informe_final.pdf
│ └── informe_tecnico.md
|
├── models/
|
├── notebooks/
│ ├── 01_eda_y_comprension_datos.ipynb
│ ├── 02_limpieza_preprocesamiento_y_particion.ipynb
│ ├── 03_modelos_ml_tradicionales.ipynb
│ └── 04_modelo_cnn.ipynb
|
├── results/
│ ├── figures/
│ └── tables/
|
├── README.md
└── requirements.txt
```

## Modelos entrenados

Por limitaciones de tamaño de GitHub, los modelos CNN entrenados (`cnn_quality.keras` y `cnn_size.keras`) no se encuentran incluidos en este repositorio, ya que cada archivo supera el límite de 100 MB permitido por la plataforma. Los notebooks incluidos en el proyecto permiten reproducir completamente el proceso de entrenamiento y generar nuevamente dichos modelos a partir de los datos originales.

Los modelos tradicionales (SVM y Random Forest) sí se encuentran disponibles dentro del repositorio debido a su menor tamaño.

## Instalación

Crear un entorno virtual:

```bash
python -m venv venv
```

Activarlo:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Ingresa a la carpeta app/ y ejecuta:

```bash
python main.py
```

Posteriormente abrir el navegador en la dirección indicada por el servidor local.

La aplicación permite:

* Cargar imágenes desde el dispositivo.
* Capturar imágenes mediante la cámara.
* Obtener predicciones de calidad y tamaño en tiempo real.

## Licencia

Este proyecto fue desarrollado con fines académicos.
