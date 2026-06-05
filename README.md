# Sistema de Clasificación Automática de Calidad y Tamaño de Frutas

## Integrantes

- Luna Catalina Martínez Vásquez
- Renzo Fernando Mosquera Daza
- Valentina Tobar Gómez

## Descripción

Este proyecto implementa un sistema de inteligencia artificial para la clasificación automática de frutas a partir de imágenes.

La solución permite predecir:

- **Calidad:** Buena, Regular o Mala.
- **Tamaño:** Pequeño, Mediano o Grande.

Para ello se evaluaron modelos de aprendizaje automático (**SVM** y **Random Forest**) y aprendizaje profundo (**CNN**), comparando su desempeño mediante diferentes métricas de clasificación.

## Dataset

Se utilizaron:

- El conjunto de datos público **Fruit Quality Classification**.
- Imágenes recolectadas y etiquetadas por los integrantes del proyecto.

## Tecnologías utilizadas

- Python
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib

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
├── reports/
│ └── figures/
|
├── README.md
└── requirements.txt
```

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

## Licencia

Este proyecto fue desarrollado con fines académicos.