| columna | descripcion |
|---|---|
| image_id | Identificador reproducible construido desde el hash de la imagen |
| source | Origen de la imagen. En esta versión usamos únicamente own |
| quality_label | Etiqueta objetivo de calidad: bad, regular o good |
| quality_label_es | Etiqueta de calidad en español |
| fruit_type | Tipo de fruta o verdura inferido desde la ruta |
| product_type | Copia de fruit_type para usar una palabra más general en el proyecto |
| size_label_manual | Etiqueta manual de tamaño si aparece en carpetas o nombres |
| relative_path | Ruta relativa a la raíz del proyecto |
| width | Ancho original en píxeles |
| height | Alto original en píxeles |
| aspect_ratio | Relación ancho/alto |
| is_square | Indica si la imagen original es cuadrada |
| requires_crop | Indica si el preprocesamiento debe ajustar la proporción |
| mode | Modo de color leído con PIL |
| file_size_kb | Peso del archivo en kilobytes |
| blur_score | Medida inicial de nitidez basada en variación de gradientes |
| brightness_mean | Brillo promedio de la imagen en escala de grises |
| brightness_std | Variación del brillo en la imagen |
| skin_tone_ratio | Proporción aproximada de píxeles con tonos similares a piel |
| border_std | Variación de color en los bordes de la imagen |
| foreground_ratio_border | Proporción aproximada de píxeles que se diferencian del borde |
| image_hash | Hash MD5 para detectar duplicados exactos |