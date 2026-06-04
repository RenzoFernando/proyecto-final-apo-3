# Diccionario de datos

| column | description |
| --- | --- |
| image_id | Identificador reproducible construido desde el hash de la imagen |
| source | Origen de la imagen: public u own |
| quality_label | Etiqueta objetivo de calidad: bad, regular o good |
| quality_label_es | Etiqueta de calidad en español |
| fruit_type | Tipo de fruta o verdura inferido desde la ruta |
| product_type | Copia de fruit_type para usar una palabra más general en el proyecto |
| size_label_manual | Etiqueta manual de tamaño si aparece en carpetas o nombres |
| relative_path | Ruta relativa a la raíz del proyecto |
| width | Ancho original en píxeles |
| height | Alto original en píxeles |
| aspect_ratio | Relación ancho/alto |
| is_square | Indica si la imagen original ya es cuadrada |
| requires_crop | Indica si la imagen requiere ajuste cuadrado en el preprocesamiento |
| mode | Modo de color original según PIL |
| file_size_kb | Peso del archivo en kilobytes |
| image_hash | Hash MD5 para detectar duplicados exactos |
