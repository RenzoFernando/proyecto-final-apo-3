# Informe Técnico Detallado

**Integrantes:** Luna Catalina Martínez Vásquez · Renzo Fernando Mosquera Daza · Valentina Tobar Gómez

---

# 1. Análisis ético

La inteligencia artificial ofrece oportunidades para automatizar tareas de inspección y clasificación de productos agrícolas. Sin embargo, su implementación también implica responsabilidades éticas relacionadas con la calidad de los datos, la confiabilidad de las predicciones y las consecuencias que pueden derivarse de las decisiones tomadas a partir de sus resultados.

En este proyecto se analizaron diferentes situaciones potencialmente problemáticas tomando como referencia el Código de Ética del IEEE [1]. El objetivo no fue únicamente desarrollar un modelo con buen desempeño, sino también comprender los posibles efectos que su uso podría tener sobre productores, comerciantes y usuarios finales.

## Tabla 1. Situaciones éticas identificadas y acciones adoptadas

| Situación o dilema                                | Principio IEEE relacionado                                | Riesgo asociado                                                                                 | Acción adoptada                                                                           |
| ------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Clasificación incorrecta de frutas                | Responsabilidad (1), Honestidad (3), Evitar daños (9)     | Pérdidas económicas por descarte de productos aptos o comercialización de productos defectuosos | El sistema se plantea como herramienta de apoyo y no como mecanismo de decisión autónoma. |
| Sesgo por datos poco representativos              | Responsabilidad (1), Honestidad (3), Trato equitativo (8) | Menor precisión para ciertas variedades de frutas o condiciones de captura                      | Complementación del dataset con imágenes propias y análisis de distribución de clases.    |
| Exceso de confianza en las predicciones           | Honestidad (3), Comprensión de consecuencias (5)          | Interpretación incorrecta de las capacidades reales del modelo                                  | Reporte transparente de métricas y documentación de limitaciones.                         |
| Uso de datasets, librerías y recursos de terceros | Reconocimiento de contribuciones (7)                      | Uso indebido de recursos o falta de atribución académica                                        | Citación adecuada de datasets, bibliografía y herramientas utilizadas.                    |

## Reflexión

El principal riesgo que logramos identificar corresponde al sesgo de los datos, ya que Colombia posee una gran diversidad de frutas, variedades y condiciones de producción que difícilmente pueden representarse completamente en un conjunto de datos limitado. Y como consecuencia, un modelo que funciona adecuadamente sobre ciertas frutas podría presentar un desempeño inferior sobre otras menos representadas.

Otro aspecto relevante corresponde a los errores de clasificación. Aunque los modelos desarrollados alcanzan niveles de desempeño elevados, ninguna solución basada en aprendizaje automático es completamente infalible. Una clasificación errónea puede traducirse en pérdidas económicas, desperdicio de alimentos o decisiones comerciales inadecuadas.

Por esta razón, consideramos que la inteligencia artificial debe utilizarse como una herramienta de apoyo a la toma de decisiones y no como un reemplazo total del criterio humano. Además, resulta fundamental comunicar de forma transparente las capacidades y limitaciones del sistema para evitar expectativas irreales sobre su funcionamiento.

Las situaciones analizadas muestran que la responsabilidad ética en sistemas de inteligencia artificial no depende únicamente del desempeño del modelo, sino también de la calidad de los datos, la transparencia de sus limitaciones y el uso adecuado de los resultados generados.

---

# 2. Análisis de impactos

El automatizar este tipo de clasificación tiene el potencial de generar beneficios importantes en la clasificación de frutas, aunque también puede producir impactos que deben ser considerados desde diferentes dimensiones.

## Tabla 2. Matriz de impactos de la solución

| Área      | Impacto positivo                                                                          | Impacto negativo                                                    | Medida de mitigación                                           |
| --------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------- |
| Social    | Mayor consistencia y objetividad en la clasificación de frutas                            | Errores o sesgos que afecten determinadas categorías o productores  | Supervisión humana y mejora continua del conjunto de datos     |
| Económica | Reducción de pérdidas por clasificaciones incorrectas y mejora de la eficiencia operativa | Costos de implementación y mantenimiento                            | Uso de herramientas de código abierto y hardware de bajo costo |
| Ambiental | Disminución potencial del desperdicio de alimentos                                        | Consumo energético asociado al entrenamiento y ejecución de modelos | Optimización de modelos y uso bajo demanda                     |
| Global    | Posibilidad de adaptación a distintos contextos agrícolas                                 | Diferencias regionales en variedades y criterios de calidad         | Reentrenamiento y validación con datos locales                 |

## Impacto social

La clasificación automática puede reducir la subjetividad presente en los procesos manuales y favorecer una evaluación más consistente de los productos. Sin embargo, errores sistemáticos o sesgos en los datos podrían afectar a ciertos productores o variedades de frutas. Por esta razón, la supervisión humana continúa siendo un elemento fundamental del proceso.

## Impacto económico

Una clasificación más precisa puede contribuir a disminuir pérdidas asociadas al descarte incorrecto de productos y mejorar la eficiencia de las actividades de selección y comercialización. No obstante, la adopción de estas tecnologías implica costos relacionados con infraestructura, mantenimiento y actualización de los modelos.

## Impacto ambiental

Una mejor clasificación de frutas puede contribuir a reducir el desperdicio alimentario y por tanto, el uso innecesario de recursos asociados a la producción agrícola. Sin embargo, los modelos de inteligencia artificial también requieren recursos computacionales y consumo energético, especialmente durante la fase de entrenamiento.

## Impacto global

El enfoque utilizado puede adaptarse a diferentes regiones y cadenas de suministro agrícola. No obstante, los criterios de calidad pueden variar entre países o mercados, por lo que cualquier implementación en nuevos contextos requiere validación y ajuste con datos locales.

## Reflexión

La inteligencia artificial puede aportar beneficios significativos en eficiencia, estandarización y reducción de desperdicios dentro de los procesos de clasificación de frutas. Sin embargo, estos beneficios dependen de una implementación responsable que considere las limitaciones de los modelos, la calidad de los datos y los posibles impactos sociales, económicos y ambientales de la tecnología.

---

# 3. Fundamentación matemática

## Formulación del problema

El problema abordado consiste en clasificar imágenes de frutas según dos variables objetivo: calidad (buena, regular o mala) y tamaño (pequeño, mediano o grande). Desde una perspectiva de aprendizaje supervisado, el objetivo es encontrar una función:

$$f(x) → y$$

donde *x* representa una imagen de entrada y *y* la categoría asociada.

Se trata de un problema complejo de ingeniería porque las características visuales que determinan la calidad o el tamaño de una fruta presentan variaciones relacionadas con iluminación, ángulo de captura, color, forma, textura y condiciones de adquisición de las imágenes.

## Construcción de la variable tamaño

El conjunto de datos original no contenía etiquetas de tamaño, por lo que fue necesario construir esta variable a partir de las imágenes disponibles. Para ello se estimó inicialmente el área ocupada por la fruta mediante segmentación basada en color. La separación entre fruta y fondo se realizó utilizando la distancia euclidiana en el espacio RGB de tal manera:

$$D = √[(R-R_f)² + (G-G_f)² + (B-B_f)²]$$

donde $R_f,G_f,B_f$ representa el color de referencia del fondo. Posteriormente se calculó un diámetro equivalente:

$$D = 2√(A/π)$$

siendo *A* el área segmentada. Finalmente, los valores obtenidos se dividieron mediante terciles para generar las categorías pequeño, mediano y grande.

## Normalización de imágenes

Antes del entrenamiento, todas las imágenes fueron redimensionadas a 128 × 128 píxeles y normalizadas mediante:

$$x_{norm} = x / 255$$

Esta transformación reduce diferencias de escala entre los valores de entrada y favorece la estabilidad numérica durante el entrenamiento de los modelos.

## Justificación de los modelos seleccionados

Se seleccionaron tres enfoques diferentes con el objetivo de comparar estrategias de complejidad creciente para la clasificación de imágenes.

### Support Vector Machine (SVM)

SVM fue elegido por ser uno de los algoritmos clásicos más utilizados en problemas de clasificación supervisada. Su capacidad para construir fronteras de decisión no lineales mediante kernels permite obtener buenos resultados incluso cuando el número de muestras es limitado.

Además, constituye una referencia importante para comparar el desempeño de modelos más complejos.

### Random Forest

Random Forest fue seleccionado por su capacidad para combinar múltiples árboles de decisión y reducir problemas de sobreajuste mediante agregación de modelos.

Este algoritmo suele ofrecer resultados robustos en conjuntos de datos heterogéneos y permite evaluar si las relaciones presentes en las características extraídas pueden modelarse adecuadamente mediante estructuras basadas en reglas.

### Redes Neuronales Convolucionales (CNN)

Las CNN fueron incluidas debido a que representan el estado del arte para tareas de clasificación de imágenes.

A diferencia de SVM y Random Forest, las CNN aprenden automáticamente características visuales relevantes como bordes, texturas, formas y patrones de color directamente desde los píxeles de entrada, evitando la necesidad de diseñar manualmente dichas características.

Por esta razón se esperaba que las CNN presentaran una mejor capacidad de generalización para la tarea propuesta.

## Métricas de evaluación

El desempeño de los modelos se evaluó mediante Accuracy, Precision, Recall y F1-Score. El uso conjunto de estas métricas permite analizar no solo la proporción global de aciertos, sino también la capacidad del modelo para identificar correctamente cada categoría y equilibrar falsos positivos y falsos negativos.

La selección de F1-Score como métrica principal resulta especialmente útil cuando existen diferencias en la cantidad de muestras por categoría.

## Interpretación de los resultados

Los resultados obtenidos muestran que las **Redes Neuronales Convolucionales** alcanzaron el mejor desempeño general en ambas tareas de clasificación. Este comportamiento se esperaba debido a que las CNN pueden aprender representaciones visuales complejas directamente desde las imágenes, mientras que los modelos tradicionales dependen en mayor medida de las características suministradas durante el entrenamiento.

La comparación entre SVM, Random Forest y CNN permitió evidenciar cómo el incremento en la capacidad de representación del modelo influye directamente en la precisión alcanzada para problemas de anotación por video.


---

### Referencias

[1] “Código de ética de IEEE.” Available: https://edu.ieee.org/ec-ups/wp-content/uploads/sites/266/CODIGO_DE_ETICA_IEEE.pdf 

 
