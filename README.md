# ECGFounder — Visualización del Espacio Latente

**Proyecto de grado · Universidad Militar Nueva Granada · Grupo GIM**  
**Autor:** Guillermo Andres Campo Benjumea 
**Director:** Prof. Wilson J. Sarmiento

Visualización interactiva 2D/3D del espacio latente del modelo ECGFounder (1024D),
coloreado por las 5 superclases diagnósticas del dataset PTB-XL.

---

## Estructura de carpetas requerida

Antes de ejecutar, verifica que el proyecto tenga esta estructura:

```
ecg-visualizacion/
├── visualize_superclasses.py     ← script principal
├── requirements.txt              ← dependencias
├── README.md                     ← este archivo
│
├── res/
│   └── embeddings - porfavor descargar desde el drive ya que es demasiado grande para el github - https://drive.google.com/drive/folders/15h1tc2VUHEynLniLuVcCxqm_eCQD4OEH?usp=sharing
│   └── projections_no_prereduction/
│       ├── proj_umap_2d_no_prereduction.npy
│       ├── proj_umap_3d_no_prereduction.npy
│       ├── proj_tsne_2d_no_prereduction.npy
│       ├── proj_tsne_3d_no_prereduction.npy
│       ├── proj_rpca_2d_no_prereduction.npy
│       ├── proj_rpca_3d_no_prereduction.npy
│       ├── proj_pacmap_2d_no_prereduction.npy
│       ├── proj_pacmap_3d_no_prereduction.npy
│       ├── proj_trimap_2d_no_prereduction.npy
│       ├── proj_trimap_3d_no_prereduction.npy
│       ├── proj_phate_2d_no_prereduction.npy
│       └── proj_phate_3d_no_prereduction.npy
│
├── csv/
│   └── ptbxl_label.csv
└── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ - porfavor descargar desde el drive ya que es demasiado grande para el github - https://drive.google.com/drive/folders/15h1tc2VUHEynLniLuVcCxqm_eCQD4OEH?usp=sharing o descargar de la pagina de physionet: https://www.physionet.org/content/ptb-xl/1.0.3/
    ├── ptbxl_database.csv
    └── scp_statements.csv
```

> **Nota:** La visualización carga automáticamente las técnicas cuyos archivos `.npy`
> estén presentes. Si solo se tienen algunas técnicas, el visualizador mostrará
> únicamente las disponibles.

---

## Requisitos

- **Python 3.9 o superior**
- Conexión a internet la primera vez (para cargar Three.js desde CDN en modo 3D)

---

## Instalación y ejecución

### Paso 1 — Clonar o descomprimir el proyecto

Coloca la carpeta del proyecto en cualquier ubicación de tu PC.

### Paso 2 — Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3 — Ejecutar la visualización

```bash
python visualize_superclasses.py
```

Verás en la terminal algo como:

```
=======================================================
  Técnicas cargadas: ['umap', 'tsne', 'rpca', ...]
  Muestras totales:  21,799
  Abre en:           http://127.0.0.1:8050
=======================================================
```

### Paso 5 — Abrir en el navegador

Abre tu navegador y ve a:

```
http://127.0.0.1:8050
```

---

## Cómo usar la visualización

| Control | Acción |
|---|---|
| **Técnica** | Selector de radio: UMAP, t-SNE, RPCA, PaCMAP, TriMAP, PHATE |
| **Dimensión** | 2D o 3D |
| **Scroll** | Zoom in/out |
| **Click + arrastrar** | Rotar (3D) / Pan (2D) |
| **Hover sobre puntos** | Ver superclase y número de muestra |
| **Vista panorámica** | Minimap en la esquina derecha — muestra el cuadrado blanco de la región visible al hacer zoom |

### Superclases (colores)

| Color | Superclase | Descripción |
|---|---|---|
| 🔵 Azul | NORM | ECG Normal |
| 🔴 Rojo | MI | Infarto de Miocardio |
| 🟠 Naranja | STTC | Cambios ST/T |
| 🟢 Verde | CD | Trastorno de Conducción |
| 🟣 Morado | HYP | Hipertrofia |
| ⚫ Gris | UNKNOWN | Sin clasificar |

---

## Solución de problemas

**Error: "No se encontró ptbxl_database.csv"**  
→ Verifica que los archivos CSV estén en `ptb-xl-data/` dentro de la carpeta del proyecto.

**Error: "No se encontraron proyecciones"**  
→ Verifica que los archivos `.npy` estén en `res/projections_no_prereduction/`.

**La página no carga en el navegador**  
→ Espera unos segundos después de ejecutar el script. El servidor tarda ~5-10 segundos en iniciar.

**El minimap 3D no muestra la nube de puntos**  
→ Requiere conexión a internet para cargar Three.js. Verifica tu conexión.

**Puerto 8050 ocupado**  
→ Cambia el puerto al final del script: `app.run(debug=False, port=8051)`

---

## Dependencias principales

| Librería | Uso |
|---|---|
| `dash` | Framework de la aplicación web interactiva |
| `plotly` | Gráficos 2D/3D interactivos |
| `dash-bootstrap-components` | Estilos y layout |
| `numpy` | Carga de proyecciones `.npy` |
| `pandas` | Lectura de archivos CSV |
| `three.js` (CDN) | Minimap 3D (se carga automáticamente) |
