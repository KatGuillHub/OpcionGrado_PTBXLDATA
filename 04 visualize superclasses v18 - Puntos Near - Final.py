"""
visualize_superclasses_v7.py
----------------------------
Minimap Three.js:
  - La cámara ROTA junto con Plotly (replica eye con zoom-out x4)
  - Dibuja un cuadrado blanco que representa la región visible del plot principal
  - El cuadrado siempre está "delante" de la cámara del minimap, en el espacio 3D
  - Sin marcador de posición, sin top-down fijo

Cómo funciona el cuadrado:
  Plotly cámara ortográfica: el frustum tiene half-size = BASE / zoom
  En Three.js, dibujamos ese rectángulo en el plano perpendicular a la
  dirección de visión, posicionado en el "centro" de la nube.
  Cuando rota → el rectángulo rota con la cámara.
  Cuando zoom → el rectángulo encoge/crece.
  Cuando pan → el rectángulo se desplaza.
  Resultado: igual que el cuadrado blanco en 2D pero en 3D.
"""

import numpy as np
import pandas as pd
import os
import ast
import csv
import json
import threading
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, Patch, no_update, clientside_callback
import dash_bootstrap_components as dbc
import wfdb
from sklearn.neighbors import NearestNeighbors

# ── Sistema de Logging de Usuarios ────────────────────────────────────────────
import atexit, signal

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
Path(LOG_DIR).mkdir(exist_ok=True)

USER_ID_FILE = os.path.join(LOG_DIR, 'user_counter.txt')
_log_lock = threading.Lock()

# Estado global de sesión activa (accesible por atexit/signal)
_active_session = {'user_id': None, 'session_start': None}

def _get_next_user_id():
    """Lee y actualiza el contador de usuarios de forma thread-safe."""
    with _log_lock:
        if os.path.exists(USER_ID_FILE):
            with open(USER_ID_FILE, 'r') as f:
                uid = int(f.read().strip()) + 1
        else:
            uid = 1
        with open(USER_ID_FILE, 'w') as f:
            f.write(str(uid))
    return uid

LOG_FIELDNAMES = [
    'user_id', 'session_start', 'session_end', 'session_duration_s',
    'event_type', 'timestamp', 'x_coord', 'y_coord',
    'method', 'dims', 'superclass_visibility', 'detail'
]

def _get_log_path(user_id):
    return os.path.join(LOG_DIR, f'user_{user_id:04d}.csv')

def _init_log(user_id, session_start):
    """Crea el CSV del usuario con cabecera."""
    path = _get_log_path(user_id)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        writer.writeheader()
    print(f'[LOG] Usuario #{user_id:04d} iniciado — {path}')
    return path

def _append_event(user_id, session_start, event_type, method='', dims='',
                  x_coord='', y_coord='', superclass_visibility='', detail='', session_end='', session_duration_s=''):
    """Agrega una fila al CSV del usuario."""
    path = _get_log_path(user_id)
    row = {
        'user_id':                f'{user_id:04d}',
        'session_start':          session_start,
        'session_end':            session_end,
        'session_duration_s':     session_duration_s,
        'event_type':             event_type,
        'timestamp':              datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        'x_coord':                x_coord,
        'y_coord':                y_coord,
        'method':                 method,
        'dims':                   dims,
        'superclass_visibility':  superclass_visibility,
        'detail':                 detail,
    }
    with _log_lock:
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
            writer.writerow(row)

def _close_session():
    """Escribe session_end en el CSV cuando el proceso termina (Ctrl+C o cierre)."""
    uid   = _active_session.get('user_id')
    start = _active_session.get('session_start')
    if not uid or not start:
        return
    now = datetime.now()
    end_str = now.strftime('%Y-%m-%d %H:%M:%S')
    try:
        start_dt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
        duration = round((now - start_dt).total_seconds(), 1)
    except Exception:
        duration = ''
    _append_event(uid, start, 'session_end',
                  session_end=end_str, session_duration_s=str(duration),
                  detail='Sesión cerrada (Ctrl+C / proceso terminado)')
    print(f'\n[LOG] Sesión #{uid:04d} cerrada — duración: {duration}s')

def _signal_handler(sig, frame):
    raise SystemExit(0)

atexit.register(_close_session)
signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ── Crear sesión al arrancar el script ────────────────────────────────────────
_SESSION_UID   = _get_next_user_id()
_SESSION_START = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
_active_session['user_id']      = _SESSION_UID
_active_session['session_start'] = _SESSION_START
_init_log(_SESSION_UID, _SESSION_START)
_append_event(_SESSION_UID, _SESSION_START, 'session_start',
              detail=f'Visualizador iniciado')

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE      = r'C:\Users\Usuario\Documents\VS Code\ECGFounder-master Github'
# --------------
# PARA OBTENER LAS PROYECCIONS, CAMBIAR SEGUN SEA LO NECESARIO
#PROJ_DIR  = os.path.join(BASE, 'res', 'projections')
PROJ_DIR = os.path.join(BASE, 'res', 'projections_no_prereduction') #1024D -> 2D/3D directo, sin PCA previo.
# --------------
PTBXL_DIR = os.path.join(BASE, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
DB_CSV    = os.path.join(PTBXL_DIR, 'ptbxl_database.csv')
SCP_CSV   = os.path.join(PTBXL_DIR, 'scp_statements.csv')
LABEL_CSV = os.path.join(BASE, 'csv', 'ptbxl_label.csv')

# ── Proyecciones ───────────────────────────────────────────────────────────────
print('Cargando proyecciones...')
ALL_METHODS = {
    'umap':'UMAP','tsne':'t-SNE','rpca':'RPCA',
    'pacmap':'PaCMAP','trimap':'TriMAP','phate':'PHATE',
}
projections = {}
available_methods = []
for key, label in ALL_METHODS.items():
    # ── Cargar proyecciones con pca ───────────────────────────────────────────
    #p2 = os.path.join(PROJ_DIR, f'proj_{key}_2d.npy') #aqui esta tomando de 1024D -> 50D -> 2D/3D
    #p3 = os.path.join(PROJ_DIR, f'proj_{key}_3d.npy')
    # ── Cargar proyecciones sin pca ───────────────────────────────────────────
    p2 = os.path.join(PROJ_DIR, f'proj_{key}_2d_no_prereduction.npy') #aqui esta tomando de 1024D -> 2D/3D
    p3 = os.path.join(PROJ_DIR, f'proj_{key}_3d_no_prereduction.npy')
    # ───────────────────────────────────────────────────────────────────────────
    if os.path.exists(p2) or os.path.exists(p3):
        available_methods.append(key)
        if os.path.exists(p2): projections[f'{key}_2d'] = np.load(p2); print(f'  ✓ {label} 2D')
        if os.path.exists(p3): projections[f'{key}_3d'] = np.load(p3); print(f'  ✓ {label} 3D')
default_method = available_methods[0]

# ── Embeddings + KNN ──────────────────────────────────────────────────────────
EMB_PATH = os.path.join(BASE, 'res', 'embeddings', 'embeddings.npy')
print('Cargando embeddings para KNN...')
embeddings = np.load(EMB_PATH)   # (21799, 1024)
knn_model  = NearestNeighbors(n_neighbors=3, metric='euclidean', algorithm='auto', n_jobs=-1)
knn_model.fit(embeddings)
print(f'  KNN listo — {embeddings.shape}')

# ── Superclases ────────────────────────────────────────────────────────────────
print('Cargando superclases...')
label_df = pd.read_csv(LABEL_CSV).dropna(subset=['filename_hr','label']).reset_index(drop=True)
db_df    = pd.read_csv(DB_CSV, index_col='ecg_id')
scp_df   = pd.read_csv(SCP_CSV, index_col=0)
scp_diag = scp_df[scp_df['diagnostic'] == 1.0]

def get_superclass_from_scp(scp_str):
    if pd.isna(scp_str): return 'UNKNOWN'
    try:
        d = ast.literal_eval(scp_str)
        best_sc, best_val = 'UNKNOWN', -1
        for code, val in d.items():
            if code in scp_diag.index:
                sc = scp_diag.loc[code, 'diagnostic_class']
                if pd.notna(sc) and sc != '' and float(val) >= best_val:
                    best_val, best_sc = float(val), sc
        return best_sc
    except: return 'UNKNOWN'

db_df['superclass']    = db_df['scp_codes'].apply(get_superclass_from_scp)
label_df['ecg_id']     = label_df['filename_hr'].apply(
    lambda x: int(os.path.basename(str(x)).replace('_hr','')))
label_df['superclass'] = label_df['ecg_id'].map(db_df['superclass']).fillna('UNKNOWN')
superclasses_array     = label_df['superclass'].values

# ── Paleta ─────────────────────────────────────────────────────────────────────
SUPERCLASS_CONFIG = {
    'NORM':   {'color':'#2196f3','label':'NORM — Normal ECG',          'opacity':0.65},
    'MI':     {'color':'#e63946','label':'MI — Myocardial Infarction', 'opacity':0.75},
    'STTC':   {'color':'#ff9800','label':'STTC — ST/T Change',         'opacity':0.75},
    'CD':     {'color':'#4caf50','label':'CD — Conduction Disturbance','opacity':0.75},
    'HYP':    {'color':'#9c27b0','label':'HYP — Hypertrophy',          'opacity':0.75},
    'UNKNOWN':{'color':'#607d8b','label':'Sin clasificar',             'opacity':0.3},
}
ORDER = ['NORM','MI','STTC','CD','HYP','UNKNOWN']
MINIMAP_COLORS = [SUPERCLASS_CONFIG.get(sc,SUPERCLASS_CONFIG['UNKNOWN'])['color']
                  for sc in superclasses_array]
LEGEND_STYLE = dict(font=dict(size=11,color='#ecf0f1'),
                    bgcolor='rgba(0,0,0,0.55)',
                    bordercolor='rgba(255,255,255,0.1)',borderwidth=1)

# ── Submuestra para Three.js ──────────────────────────────────────────────────
MINIMAP_MAX = 4000
_rng = np.random.default_rng(42)

def _subsample_for_threejs(method):
    key = f'{method}_3d'
    if key not in projections: key = f'{available_methods[0]}_3d'
    proj = projections[key]
    n    = len(proj)
    idx  = _rng.choice(n, min(MINIMAP_MAX, n), replace=False)
    pts  = proj[idx]
    mn, mx_ = pts.min(0), pts.max(0)
    rng_    = np.where((mx_-mn) > 1e-9, mx_-mn, 1.0)
    pts_n   = 2.0*(pts-mn)/rng_ - 1.0
    sc_sub  = superclasses_array[idx]
    colors  = [SUPERCLASS_CONFIG.get(sc,SUPERCLASS_CONFIG['UNKNOWN'])['color'] for sc in sc_sub]
    return {'x':pts_n[:,0].tolist(),'y':pts_n[:,1].tolist(),
            'z':pts_n[:,2].tolist(),'colors':colors}

_threejs_data = {}
for m in available_methods:
    if f'{m}_3d' in projections:
        _threejs_data[m] = _subsample_for_threejs(m)


# ══════════════════════════════════════════════════════════════════════════════
# THREE.JS HTML
# ══════════════════════════════════════════════════════════════════════════════
THREEJS_HTML = """
<!DOCTYPE html>
<html style="margin:0;padding:0;background:#0f3460;">
<body style="margin:0;padding:0;overflow:hidden;background:#0f3460;">
<canvas id="c" style="width:100%;height:100%;display:block;"></canvas>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
/*
CORRECCIONES v9
===============
1. ESPEJADO: Al hacer swap Plotly(x,y,z)→Three(x,z,y) se invierte la
   "mano" del sistema de coordenadas. La corrección es negar X:
   Plotly(x, y, z) → Three(-x, z, y)
   Esto hace que la rotación orbital coincida con Plotly.

2. ZOOM: Plotly ortográfico NO mueve el eye al hacer scroll — usa un
   parámetro interno que no expone en relayoutData. La solución es
   leer el evento 'wheel' directamente en el canvas del iframe padre
   y acumular el factor de zoom localmente. El zoom acumulado escala
   el cuadrado sin depender de Plotly.
   Adicionalmente, cuando Plotly sí manda scene.camera (al soltar el
   mouse), lo usamos para sincronizar posición y orientación.
*/
// ─────────────────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
canvas.width  = canvas.parentElement ? canvas.parentElement.clientWidth  : 320;
canvas.height = canvas.parentElement ? canvas.parentElement.clientHeight : 180;
const W = canvas.width || 320;
const H = canvas.height || 180;

const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setSize(W, H);
renderer.setClearColor(0x0f3460, 1);

const scene = new THREE.Scene();

const aspect   = W / H;
const VIEW     = 1.2;   // frustum más cerrado → nube ocupa más del minimap
const ZOOM_OUT = 2.2;   // cámara más cerca → nube más grande en pantalla
const cam = new THREE.OrthographicCamera(
    -VIEW*aspect, VIEW*aspect, VIEW, -VIEW, 0.01, 100
);
// Posición inicial: eye default de Plotly (1.5,1.5,1.5) con swap correcto
// FIX ESPEJADO: Plotly(x,y,z) → Three(-x, z, y)  (negar X para corregir handedness)
cam.position.set(-1.5*ZOOM_OUT, 1.5*ZOOM_OUT, 1.5*ZOOM_OUT);
cam.lookAt(0, 0, 0);

// ── Acumulador de zoom local ──────────────────────────────────────────────────
// Plotly ortográfico NO expone el zoom en relayoutData cuando se hace scroll.
// Lo capturamos localmente escuchando el evento 'wheel' en el documento padre
// via postMessage, o directamente aquí si tenemos acceso.
// El zoom se acumula multiplicativamente: >1 = zoom in, <1 = zoom out.
let localZoom = 1.0;

// Escuchar scroll del iframe para acumular zoom
// (el iframe recibe eventos de scroll propios, no del plot principal)
// En cambio recibimos el zoom via window.name cuando Plotly lo manda
// y también escuchamos mensajes postMessage del padre
window.addEventListener('message', function(e) {
    try {
        const d = typeof e.data === 'string' ? JSON.parse(e.data) : e.data;
        if (d && d.type === 'zoom') {
            localZoom = d.zoom;
            drawRect();
        }
    } catch(e) {}
});

// ── Nube de puntos ───────────────────────────────────────────────────────────
let cloud = null;

function buildCloud(d) {
    if (cloud) { scene.remove(cloud); cloud.geometry.dispose(); cloud.material.dispose(); }
    const n   = d.x.length;
    const pos = new Float32Array(n*3);
    const col = new Float32Array(n*3);
    const c   = new THREE.Color();
    for (let i=0; i<n; i++) {
        // FIX ESPEJADO: Plotly(x,y,z) → Three(-x, z, y)
        pos[i*3]   = -d.x[i];   // negar X corrige el espejo
        pos[i*3+1] =  d.z[i];   // z_plotly → Y three.js (arriba)
        pos[i*3+2] =  d.y[i];   // y_plotly → Z three.js
        c.set(d.colors[i]);
        col[i*3]=c.r; col[i*3+1]=c.g; col[i*3+2]=c.b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
    geo.setAttribute('color',    new THREE.BufferAttribute(col,3));
    cloud = new THREE.Points(geo, new THREE.PointsMaterial({
        size: 0.04, vertexColors: true, opacity: 0.55, transparent: true
    }));
    scene.add(cloud);
}

// ── Cuadrado blanco ──────────────────────────────────────────────────────────
const planeGeo  = new THREE.PlaneGeometry(1, 1);
const edgesGeo  = new THREE.EdgesGeometry(planeGeo);
const border    = new THREE.LineSegments(edgesGeo,
                    new THREE.LineBasicMaterial({color:0xffffff, linewidth:2}));
border.visible  = false;
scene.add(border);

const fill = new THREE.Mesh(planeGeo.clone(),
                new THREE.MeshBasicMaterial({
                    color:0xffffff, transparent:true, opacity:0.06,
                    side:THREE.DoubleSide}));
fill.visible = false;
scene.add(fill);

// Estado actual de la cámara (se actualiza con cada mensaje)
let curEye    = {x:1.5,  y:1.5, z:1.5};
let curCenter = {x:0,    y:0,   z:0};

// BASE_HALF: a zoom=1, el cuadrado tiene radio 1.0 en espacio normalizado
const BASE_HALF = 0.65;  // a zoom=1 el cuadrado cubre bien la nube en la vista acercada

// ── Dibujar cuadrado con estado actual ───────────────────────────────────────
function drawRect() {
    // Tamaño inversamente proporcional al zoom acumulado
    const size = (BASE_HALF * 2) / localZoom;

    // Centro: el "center" de Plotly con swap correcto (-x, z, y)
    const cx = curCenter.x != null ? curCenter.x : 0;
    const cy = curCenter.y != null ? curCenter.y : 0;
    const cz = curCenter.z != null ? curCenter.z : 0;
    const centerV = new THREE.Vector3(-cx, cz, cy);

    border.position.copy(centerV);
    border.scale.set(size, size, 1);
    fill.position.copy(centerV);
    fill.scale.set(size, size, 1);

    // lookAt hacia la cámara → cuadrado siempre perpendicular a la visión
    border.lookAt(cam.position);
    fill.lookAt(cam.position);

    border.visible = true;
    fill.visible   = true;
}

// ── Actualizar cámara ─────────────────────────────────────────────────────────
function updateCamera(eye, center) {
    curEye    = eye;
    curCenter = center;

    const ex = eye.x != null ? eye.x : 1.5;
    const ey = eye.y != null ? eye.y : 1.5;
    const ez = eye.z != null ? eye.z : 1.5;
    const cx = center.x != null ? center.x : 0;
    const cy = center.y != null ? center.y : 0;
    const cz = center.z != null ? center.z : 0;

    const eyeDist = Math.sqrt(ex*ex + ey*ey + ez*ez) || 2.598;

    // FIX ESPEJADO: negar X en la posición de la cámara también
    cam.position.set(
        -(ex / eyeDist) * 2.598 * ZOOM_OUT,
         (ez / eyeDist) * 2.598 * ZOOM_OUT,
         (ey / eyeDist) * 2.598 * ZOOM_OUT
    );
    // lookAt al center con swap correcto
    cam.lookAt(-cx, cz, cy);
    cam.up.set(0, 1, 0);
    cam.updateMatrixWorld();

    drawRect();
}

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, cam);
}
animate();

// ── Polling (window.name, cada 50ms) ─────────────────────────────────────────
let lastMsg = '';

setInterval(function() {
    try {
        const raw = window.name;
        if (!raw || raw === lastMsg) return;
        lastMsg = raw;
        const msg = JSON.parse(raw);

        if ((msg.type === 'init' || msg.type === 'cloud') && msg.data) {
            buildCloud(msg.data);
            // Reset completo al cambiar de técnica
            localZoom = 1.0;
            curEye    = {x:1.5, y:1.5, z:1.5};
            curCenter = {x:0,   y:0,   z:0};
            // Resetear cámara del minimap a posición default
            cam.position.set(-1.5*ZOOM_OUT, 1.5*ZOOM_OUT, 1.5*ZOOM_OUT);
            cam.lookAt(0, 0, 0);
            cam.updateMatrixWorld();
            border.visible = false;
            fill.visible   = false;
            // Pedir al parent que resetee _zoomAccum via postMessage
            try { window.parent.postMessage({type:'resetZoom'}, '*'); } catch(e) {}
        }

        if (msg.type === 'camera') {
            const eye    = msg.eye    || {x:1.5, y:1.5, z:1.5};
            const center = msg.center || {x:0,   y:0,   z:0};
            // Si Plotly manda projection.scale explícito, úsalo
            if (msg.zoom && msg.zoom !== 1.0) localZoom = msg.zoom;
            updateCamera(eye, center);
        }

        if (msg.type === 'zoom') {
            localZoom = msg.zoom || 1.0;
            drawRect();
        }
    } catch(e) { console.warn('minimap:', e); }
}, 50);

</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS PLOTLY
# ══════════════════════════════════════════════════════════════════════════════
def make_figure(method, dims):
    key = f'{method}_{dims}d'
    if key not in projections:
        alt = f'{method}_2d' if dims==3 else f'{method}_3d'
        if alt not in projections: return go.Figure()
        key, dims = alt, (2 if dims==3 else 3)
    proj  = projections[key]
    is_3d = (dims == 3)
    traces = []
    for sc in ORDER:
        cfg  = SUPERCLASS_CONFIG.get(sc, SUPERCLASS_CONFIG['UNKNOWN'])
        mask = superclasses_array == sc
        if mask.sum() == 0: continue
        pts  = proj[mask]
        idxs = np.where(mask)[0]
        hover = [f'<b>{sc}</b><br>{cfg["label"]}<br>Muestra: {i}' for i in idxs]
        if is_3d:
            traces.append(go.Scatter3d(
                x=pts[:,0],y=pts[:,1],z=pts[:,2], mode='markers',
                name=f'{cfg["label"]} ({mask.sum():,})',
                marker=dict(size=2.5,color=cfg['color'],opacity=cfg['opacity']),
                text=hover, hoverinfo='text'))
        else:
            traces.append(go.Scattergl(
                x=pts[:,0],y=pts[:,1], mode='markers',
                name=f'{cfg["label"]} ({mask.sum():,})',
                marker=dict(size=4,color=cfg['color'],opacity=cfg['opacity']),
                text=hover, hoverinfo='text'))
    title = f'{ALL_METHODS.get(method,method)} {dims}D — Superclases PTB-XL (n={len(proj):,})'
    if is_3d:
        layout = go.Layout(
            title=dict(text=title,font=dict(size=14,color='#ecf0f1'),x=0.5),
            paper_bgcolor='#1a1a2e',
            scene=dict(
                xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showbackground=False),
                yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showbackground=False),
                zaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showbackground=False),
                bgcolor='#1a1a2e',
                camera=dict(eye=dict(x=1.5,y=1.5,z=1.5),
                            center=dict(x=0,y=0,z=0),
                            projection=dict(type='orthographic'))),
            legend=LEGEND_STYLE,
            margin=dict(l=0,r=0,t=40,b=0), height=650,
            # uirevision FIJO: Mantiene el estado de zoom/pan/rotación durante toda la sesión
            # No importa si cambias de toolkit (rotate/pan), la vista se preserva
            # Solo se resetea al cambiar de técnica (UMAP→t-SNE) o de dimensión (2D→3D)
            uirevision='3d_session')
    else:
        layout = go.Layout(
            title=dict(text=title,font=dict(size=14,color='#ecf0f1'),x=0.5),
            paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e',
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,scaleanchor='x'),
            legend=LEGEND_STYLE,
            margin=dict(l=10,r=10,t=40,b=10), height=650)
    return go.Figure(data=traces, layout=layout)


def make_minimap_2d(method):
    key = f'{method}_2d'
    if key not in projections: key = f'{available_methods[0]}_2d'
    proj = projections[key]
    return go.Figure(
        data=[go.Scattergl(x=proj[:,0],y=proj[:,1],mode='markers',
                           marker=dict(size=1.5,color=MINIMAP_COLORS,opacity=0.4),
                           hoverinfo='skip',showlegend=False)],
        layout=go.Layout(
            paper_bgcolor='#0f3460', plot_bgcolor='#0f3460',
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,
                       fixedrange=True,title=''),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,
                       fixedrange=True,scaleanchor='x',title=''),
            margin=dict(l=4,r=4,t=4,b=4), height=180, shapes=[]))


# ══════════════════════════════════════════════════════════════════════════════
# HELPER PARA VISIBILIDAD DE SUPERCLASES
# ══════════════════════════════════════════════════════════════════════════════

def _get_superclass_visibility_state(main_plot_figure):
    """
    Extrae el estado de visibilidad de las superclases desde la figura.
    Retorna string con formato: "NORM: ON, MI: OFF, STTC: ON, CD: ON, HYP: OFF, UNKNOWN: ON"
    """
    if not main_plot_figure or 'data' not in main_plot_figure:
        return ', '.join([f'{sc}: ON' for sc in ORDER])
    
    visibility_map = {}
    traces_data = main_plot_figure.get('data', [])
    
    for trace in traces_data:
        name = trace.get('name', '')
        visible = trace.get('visible', True)
        if visible is False:
            visible_str = 'OFF'
        else:
            visible_str = 'ON'
        
        for sc in ORDER:
            if sc in name:
                visibility_map[sc] = visible_str
                break
    
    result = []
    for sc in ORDER:
        state = visibility_map.get(sc, 'ON')
        result.append(f'{sc}: {state}')
    
    return ', '.join(result)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS ECG
# ══════════════════════════════════════════════════════════════════════════════

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

def _get_ecg_info(sample_idx):
    """Dado el índice en label_df, retorna (signal 12×N, metadata dict)."""
    row    = label_df.iloc[sample_idx]
    ecg_id = int(os.path.basename(str(row['filename_hr'])).replace('_hr',''))
    db_row = db_df.loc[ecg_id]
    fname  = str(db_row['filename_lr'])   # e.g. records100/00000/00001_lr
    path   = os.path.join(PTBXL_DIR, fname)
    try:
        sig, meta = wfdb.rdsamp(path)     # sig: (N, 12)
        sig = sig.T                       # → (12, N)
    except Exception as e:
        print(f'[ECG] Error leyendo {path}: {e}')
        sig = np.zeros((12, 1000))
    age  = db_row.get('age', '?')
    sex  = 'Mujer' if db_row.get('sex', 0) == 0 else 'Hombre'
    sc   = superclasses_array[sample_idx]
    return sig, {'age': age, 'sex': sex, 'superclass': sc, 'ecg_id': ecg_id, 'idx': sample_idx}


def _make_ecg_figure(sig, meta, lead_idx=1, title_prefix=''):
    """Crea figura Plotly de una derivación del ECG."""
    sc   = meta['superclass']
    cfg  = SUPERCLASS_CONFIG.get(sc, SUPERCLASS_CONFIG['UNKNOWN'])
    color = cfg['color']
    lead_sig = sig[lead_idx]
    age  = meta['age']
    age_str = f"{int(age)} años" if not (isinstance(age, float) and np.isnan(age)) else '?'
    subtitle = f"{sc}  ·  {meta['sex']}  ·  {age_str}  ·  Lead {LEAD_NAMES[lead_idx]}"
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        y=lead_sig, mode='lines',
        line=dict(color=color, width=1),
        hoverinfo='skip', showlegend=False))
    fig.update_layout(
        title=dict(text=f'<b>{title_prefix}</b>  <span style="font-size:11px;color:#aaa">{subtitle}</span>',
                   font=dict(size=12, color='#ecf0f1'), x=0, xanchor='left'),
        paper_bgcolor='#16213e', plot_bgcolor='#16213e',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=8, r=8, t=30, b=8),
        height=130)
    return fig


_EMPTY_ECG_FIG = go.Figure(layout=go.Layout(
    paper_bgcolor='#16213e', plot_bgcolor='#16213e',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    margin=dict(l=8, r=8, t=8, b=8), height=130))


# ══════════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════════
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = 'ECGFounder — Superclases PTB-XL'

CARD_STYLE = {'backgroundColor':'#16213e','border':'1px solid #0f3460',
              'borderRadius':'8px','padding':'12px'}
method_options = [{'label':f' {ALL_METHODS[m]}','value':m} for m in available_methods]

app.layout = dbc.Container([
    html.H4('ECGFounder · Superclases PTB-XL',
            style={'color':'#4caf50','fontFamily':'monospace',
                   'letterSpacing':'2px','marginTop':'18px','marginBottom':'4px'}),
    html.P('Proyección del espacio latente (1024D → 2D/3D) coloreada por superclase diagnóstica',
           style={'color':'#7f8c9a','fontFamily':'monospace',
                  'fontSize':'12px','marginBottom':'16px'}),

    dbc.Row([
        dbc.Col([
            html.Label('Técnica',style={'color':'#aab','fontSize':'12px','fontFamily':'monospace'}),
            dbc.RadioItems(id='method-selector',options=method_options,value=default_method,
                           inline=True,style={'color':'#ecf0f1','fontFamily':'monospace','fontSize':'13px'}),
        ], width='auto'),
        dbc.Col([
            html.Label('Dimensión',style={'color':'#aab','fontSize':'12px','fontFamily':'monospace'}),
            dbc.RadioItems(id='dim-selector',
                           options=[{'label':' 2D','value':2},{'label':' 3D','value':3}],
                           value=2,inline=True,
                           style={'color':'#ecf0f1','fontFamily':'monospace','fontSize':'13px'}),
        ], width='auto'),
    ], className='mb-3', style=CARD_STYLE),

    dbc.Row([
        dbc.Col([
            html.Div(style={'position':'relative'}, children=[
                dcc.Graph(id='main-plot',figure=make_figure(default_method,2),
                          config={'scrollZoom':True,'displayModeBar':True,
                                  'modeBarButtonsToRemove':['select2d','lasso2d']},
                          style={'borderRadius':'8px','overflow':'hidden'}),
                # ── Gizmo de ejes ─────────────────────────────────────────────
                html.Div(id='gizmo-container', style={
                    'position':'absolute','bottom':'36px','left':'16px',
                    'width':'80px','height':'80px','pointerEvents':'none',
                    'zIndex':'10',
                }),
            ]),
        ], width=9),

        dbc.Col([
            html.Div([
                html.P('Vista panorámica',
                       style={'color':'#7f8c9a','fontSize':'11px',
                              'fontFamily':'monospace','marginBottom':'4px'}),
                html.Div(
                    dcc.Graph(id='mini-map-2d',figure=make_minimap_2d(default_method),
                              config={'displayModeBar':False},
                              style={'borderRadius':'6px','overflow':'hidden'}),
                    id='minimap-2d-container',style={'display':'block'}),
                html.Div(
                    html.Iframe(id='minimap-3d-iframe',srcDoc=THREEJS_HTML,
                                style={'width':'100%','height':'180px','border':'none',
                                       'borderRadius':'6px','backgroundColor':'#0f3460'}),
                    id='minimap-3d-container',style={'display':'none'}),
                html.P(id='minimap-label',
                       children='Modo: 2D  |  haz zoom para ver el cuadrado',
                       style={'color':'#7f8c9a','fontSize':'10px','fontFamily':'monospace',
                              'marginTop':'4px','textAlign':'center'}),

                # ── Panel ECG ────────────────────────────────────────────────
                html.Div(id='ecg-panel', children=[
                    html.P('Selecciona un punto para ver su ECG',
                           style={'color':'#7f8c9a','fontSize':'10px',
                                  'fontFamily':'monospace','textAlign':'center',
                                  'marginTop':'8px','marginBottom':'0'}),
                ], style={'marginTop':'10px'}),
            ], style=CARD_STYLE),
        ], width=3),
    ]),

    dcc.Store(id='camera-store'),
    dcc.Store(id='dim-store',      data=2),
    dcc.Store(id='method-store',   data=default_method),
    dcc.Store(id='selected-idx',   data=None),
    dcc.Store(id='user-session',
              data={'user_id': _SESSION_UID, 'session_start': _SESSION_START}),
    dcc.Store(id='log-event-store',data=None),

    # ── Barra de estado de sesión (esquina inferior derecha) ──────────────────
    html.Div(f'👤 Usuario #{_SESSION_UID:04d}  |  ⏱ {_SESSION_START}',
             id='session-status-bar',
             style={'position':'fixed','bottom':'10px','right':'20px',
                    'backgroundColor':'rgba(15,52,96,0.85)',
                    'border':'1px solid #4caf50','borderRadius':'6px',
                    'padding':'6px 14px','fontFamily':'monospace',
                    'fontSize':'11px','color':'#4caf50','zIndex':'999'}),

], fluid=True, style={'backgroundColor':'#1a1a2e','minHeight':'100vh','padding':'0 20px 30px'})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('main-plot',           'figure'),
    Output('mini-map-2d',         'figure'),
    Output('minimap-2d-container','style'),
    Output('minimap-3d-container','style'),
    Output('minimap-label',       'children'),
    Output('dim-store',           'data'),
    Output('method-store',        'data'),
    Input('method-selector','value'),
    Input('dim-selector',   'value'),
)
def update_on_change(method, dims):
    main_fig = make_figure(method, dims)
    if dims == 2:
        return (main_fig, make_minimap_2d(method),
                {'display':'block'}, {'display':'none'},
                'Modo: 2D  |  haz zoom para ver el cuadrado', dims, method)
    else:
        return (main_fig, no_update,
                {'display':'none'}, {'display':'block'},
                'Modo: 3D  |  haz zoom/rota para ver el cuadrado', dims, method)


@app.callback(
    Output('camera-store','data'),
    Input('main-plot',   'relayoutData'),
    State('dim-store',   'data'),
    State('method-store','data'),
    prevent_initial_call=True,
)
def relay_to_store(relayout_data, dims, method):
    #print(f"KEYS: {set(relayout_data.keys()) if relayout_data else None}")
    #print(f"DATA: {relayout_data}")
    if not relayout_data or dims != 3:
        return no_update

    # Ignorar eventos que solo traen dragmode (cambio de herramienta en toolbar)
    # Estos no contienen datos de cámara y no deben enviarse al minimap
    keys = set(relayout_data.keys())
    if keys <= {'scene.dragmode', 'dragmode', 'autosize'}:
        return no_update

    cam = relayout_data.get('scene.camera')
    if cam:
        eye    = cam.get('eye',    {'x':1.5,'y':1.5,'z':1.5})
        center = cam.get('center', {'x':0,'y':0,'z':0})
        zoom   = (cam.get('projection') or {}).get('scale', 1.0) or 1.0

        # Ignorar reset interno de Plotly al cambiar de herramienta en toolbar:
        # manda eye exactamente (1.5, 1.5, 1.5) y center (0, 0, 0).
        ex = eye.get('x', 0); ey_v = eye.get('y', 0); ez = eye.get('z', 0)
        cx = center.get('x', 0); cy = center.get('y', 0); cz = center.get('z', 0)
        if (abs(cx) < 0.001 and abs(cy) < 0.001 and abs(cz) < 0.001 and
                abs(ex - 1.5) < 0.001 and abs(ey_v - 1.5) < 0.001 and abs(ez - 1.5) < 0.001):
            return no_update
    else:
        ex = relayout_data.get('scene.camera.eye.x')
        ey = relayout_data.get('scene.camera.eye.y')
        ez = relayout_data.get('scene.camera.eye.z')
        if ex is None and ey is None and ez is None:
            return no_update
        eye    = {'x': ex or 1.5, 'y': ey or 1.5, 'z': ez or 1.5}
        center = {
            'x': relayout_data.get('scene.camera.center.x', 0),
            'y': relayout_data.get('scene.camera.center.y', 0),
            'z': relayout_data.get('scene.camera.center.z', 0),
        }
        zoom = relayout_data.get('scene.camera.projection.scale', 1.0) or 1.0

    return {'type':'camera','eye':eye,'center':center,'zoom':zoom,'method':method}


# Clientside: Store → iframe + captura de zoom via scroll de Plotly
clientside_callback(
    """
    function(cameraData, dimData, methodData) {
        const iframe = document.getElementById('minimap-3d-iframe');
        if (!iframe || !iframe.contentWindow) return window.dash_clientside.no_update;

        // Mandar datos de cámara al iframe
        if (cameraData) {
            try { iframe.contentWindow.name = JSON.stringify(cameraData); } catch(e) {}
        }

        // Escuchar reset desde el iframe cuando cambia de técnica
        if (!window._zoomResetListenerAttached) {
            window._zoomResetListenerAttached = true;
            window.addEventListener('message', function(e) {
                if (e.data && e.data.type === 'resetZoom') {
                    const mp = document.getElementById('main-plot');
                    if (mp) mp._zoomAccum = 1.0;
                }
            });
        }

        // Capturar eventos de scroll del plot principal para detectar zoom
        // Plotly ortográfico no expone el zoom en relayoutData al hacer scroll,
        // pero sí actualiza internamente scene.aspectratio.
        // Lo interceptamos escuchando el evento 'plotly_relayout' en el DOM.
        const mainPlot = document.getElementById('main-plot');
        if (mainPlot && !mainPlot._zoomListenerAttached) {
            mainPlot._zoomListenerAttached = true;
            mainPlot._zoomAccum = 1.0;

            mainPlot.addEventListener('wheel', function(e) {
                // Plotly usa deltaY para zoom: negativo = zoom in
                const delta = e.deltaY || e.wheelDelta || 0;
                const factor = delta > 0 ? 0.9 : 1.1;  // igual que Plotly
                mainPlot._zoomAccum *= factor;
                mainPlot._zoomAccum = Math.max(0.05, Math.min(20, mainPlot._zoomAccum));

                try {
                    iframe.contentWindow.name = JSON.stringify({
                        type: 'zoom',
                        zoom: mainPlot._zoomAccum
                    });
                } catch(e2) {}
            }, {passive: true});
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('camera-store','data', allow_duplicate=True),
    Input('camera-store', 'data'),
    Input('dim-store',    'data'),
    Input('method-store', 'data'),
    prevent_initial_call=True,
)


@app.callback(
    Output('camera-store','data', allow_duplicate=True),
    Input('method-store','data'),
    Input('dim-store',   'data'),
    prevent_initial_call=True,
)
def send_cloud(method, dims):
    if dims != 3: return no_update
    data = _threejs_data.get(method)
    if data is None: return no_update
    return {'type':'cloud','data':data,'method':method}


@app.callback(
    Output('mini-map-2d','figure', allow_duplicate=True),
    Input('main-plot',  'relayoutData'),
    State('dim-store',  'data'),
    prevent_initial_call=True,
)
def update_rect_2d(relayout_data, dims):
    if not relayout_data or dims != 2: return no_update
    x0 = relayout_data.get('xaxis.range[0]')
    x1 = relayout_data.get('xaxis.range[1]')
    y0 = relayout_data.get('yaxis.range[0]')
    y1 = relayout_data.get('yaxis.range[1]')
    patched = Patch()
    if all(v is not None for v in [x0,x1,y0,y1]):
        patched['layout']['shapes'] = [dict(
            type='rect',x0=x0,x1=x1,y0=y0,y1=y1,
            line=dict(color='#ffffff',width=2),
            fillcolor='rgba(255,255,255,0.08)')]
    else:
        patched['layout']['shapes'] = []
    return patched


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACK: SELECCIÓN DE PUNTO → ECG + VECINOS
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('ecg-panel',    'children'),
    Output('main-plot',    'figure', allow_duplicate=True),
    Output('selected-idx', 'data'),
    Input('main-plot',     'clickData'),
    State('method-store',  'data'),
    State('dim-store',     'data'),
    State('main-plot',     'figure'),
    prevent_initial_call=True,
)
def on_point_click(click_data, method, dims, current_fig):
    if not click_data:
        return no_update, no_update, no_update

    pts = click_data.get('points', [])
    if not pts:
        return no_update, no_update, no_update

    # ── Extraer índice de muestra del hover text ──────────────────────────────
    text = pts[0].get('text', '')
    try:
        sample_idx = int(text.split('Muestra: ')[-1])
    except Exception:
        return no_update, no_update, no_update

    # ── KNN en espacio 1024D ──────────────────────────────────────────────────
    # n_neighbors=3: el primero es el propio punto, los otros 2 son los vecinos
    dists, idxs = knn_model.kneighbors([embeddings[sample_idx]])
    neighbor_idxs = [int(idxs[0][1]), int(idxs[0][2])]

    # ── Leer los 3 ECGs ───────────────────────────────────────────────────────
    sig0, meta0 = _get_ecg_info(sample_idx)
    sig1, meta1 = _get_ecg_info(neighbor_idxs[0])
    sig2, meta2 = _get_ecg_info(neighbor_idxs[1])

    fig0 = _make_ecg_figure(sig0, meta0, title_prefix='Punto seleccionado')
    fig1 = _make_ecg_figure(sig1, meta1, title_prefix='Vecino más cercano')
    fig2 = _make_ecg_figure(sig2, meta2, title_prefix='2° vecino más cercano')

    ecg_children = [
        html.Hr(style={'borderColor':'#0f3460','margin':'8px 0'}),
        dcc.Graph(figure=fig0, config={'displayModeBar':False},
                  style={'borderRadius':'6px','overflow':'hidden','marginBottom':'4px'}),
        dcc.Graph(figure=fig1, config={'displayModeBar':False},
                  style={'borderRadius':'6px','overflow':'hidden','marginBottom':'4px'}),
        dcc.Graph(figure=fig2, config={'displayModeBar':False},
                  style={'borderRadius':'6px','overflow':'hidden'}),
    ]

    # ── Marcar vecinos en el scatter ──────────────────────────────────────────
    # Reconstruimos la figura eliminando traces _sel/_nbr anteriores
    key = f'{method}_{dims}d'
    if key in projections and current_fig:
        import copy
        new_fig = copy.deepcopy(current_fig)
        new_fig['data'] = [t for t in new_fig['data']
                           if t.get('name') not in ('_sel', '_nbr')]
        proj   = projections[key]
        n_idxs = [neighbor_idxs[0], neighbor_idxs[1]]
        n_pts  = proj[n_idxs]
        sel_pt = proj[[sample_idx]]

        if dims == 3:
            new_fig['data'] += [
                go.Scatter3d(
                    x=[sel_pt[0,0]], y=[sel_pt[0,1]], z=[sel_pt[0,2]],
                    mode='markers', name='_sel', showlegend=False,
                    marker=dict(size=8, color='#ffffff', symbol='diamond',
                                line=dict(color='#ffeb3b', width=2))),
                go.Scatter3d(
                    x=n_pts[:,0], y=n_pts[:,1], z=n_pts[:,2],
                    mode='markers', name='_nbr', showlegend=False,
                    marker=dict(size=7, color='#ffeb3b', symbol='circle',
                                line=dict(color='#ffffff', width=2))),
            ]
        else:
            new_fig['data'] += [
                go.Scattergl(
                    x=[sel_pt[0,0]], y=[sel_pt[0,1]],
                    mode='markers', name='_sel', showlegend=False,
                    marker=dict(size=12, color='#ffffff', symbol='diamond',
                                line=dict(color='#ffeb3b', width=2))),
                go.Scattergl(
                    x=n_pts[:,0], y=n_pts[:,1],
                    mode='markers', name='_nbr', showlegend=False,
                    marker=dict(size=10, color='#ffeb3b', symbol='circle',
                                line=dict(color='#ffffff', width=2))),
            ]

    return ecg_children, new_fig, sample_idx


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS DE LOGGING
# ══════════════════════════════════════════════════════════════════════════════


@app.callback(
    Output('user-session','data', allow_duplicate=True),
    Input('method-selector','value'),
    Input('dim-selector',   'value'),
    State('user-session',   'data'),
    State('method-store',   'data'),
    State('dim-store',      'data'),
    prevent_initial_call=True,
)
def log_control_change(method, dims, session, prev_method, prev_dims):
    """Registra cambios de técnica o dimensión."""
    if not session:
        return no_update
    uid   = session['user_id']
    start = session['session_start']
    if method != prev_method:
        _append_event(uid, start, 'method_change',
                      method=method, dims=str(dims),
                      detail=f'{prev_method} → {method}')
    if dims != prev_dims:
        _append_event(uid, start, 'dim_change',
                      method=method, dims=str(dims),
                      detail=f'{prev_dims}D → {dims}D')
    return no_update


@app.callback(
    Output('user-session','data', allow_duplicate=True),
    Input('main-plot','relayoutData'),
    State('user-session', 'data'),
    State('method-store', 'data'),
    State('dim-store',    'data'),
    prevent_initial_call=True,
)
def log_plot_interactions(relayout_data, session, method, dims):
    """Registra zoom, pan, rotación y reset desde relayoutData de Plotly."""
    if not session or not relayout_data:
        return no_update
    uid   = session['user_id']
    start = session['session_start']
    keys  = set(relayout_data.keys())

    # ── Reset (doble click en Plotly) ─────────────────────────────────────────
    if 'xaxis.autorange' in keys or 'scene.camera' in keys:
        cam = relayout_data.get('scene.camera')
        if cam:
            eye = cam.get('eye', {})
            ex, ey, ez = eye.get('x',0), eye.get('y',0), eye.get('z',0)
            if abs(ex-1.5)<0.01 and abs(ey-1.5)<0.01 and abs(ez-1.5)<0.01:
                _append_event(uid, start, 'reset', method=method, dims=str(dims),
                              detail='Camera reset to default')
                return no_update
            
            center = cam.get('center', {})
            cx, cy, cz = center.get('x',0), center.get('y',0), center.get('z',0)
            
            if abs(cx) > 0.01 or abs(cy) > 0.01 or abs(cz) > 0.01:
                _append_event(uid, start, 'pan_3d', method=method, dims=str(dims),
                              detail=f'center=({cx:.3f},{cy:.3f},{cz:.3f})')
            else:
                _append_event(uid, start, 'rotate_3d', method=method, dims=str(dims),
                              detail=f'eye=({ex:.3f},{ey:.3f},{ez:.3f})')
            return no_update

    # ── Zoom 2D ───────────────────────────────────────────────────────────────
    if 'xaxis.range[0]' in keys:
        x0 = relayout_data.get('xaxis.range[0]', '')
        x1 = relayout_data.get('xaxis.range[1]', '')
        y0 = relayout_data.get('yaxis.range[0]', '')
        y1 = relayout_data.get('yaxis.range[1]', '')
        _append_event(uid, start, 'zoom_2d', method=method, dims=str(dims),
                      x_coord=f'{x0:.4f}:{x1:.4f}', y_coord=f'{y0:.4f}:{y1:.4f}',
                      detail='zoom/pan 2D')
        return no_update

    # ── autorange reset 2D ────────────────────────────────────────────────────
    if 'xaxis.autorange' in keys:
        _append_event(uid, start, 'reset', method=method, dims=str(dims),
                      detail='autorange reset 2D')

    return no_update


@app.callback(
    Output('user-session','data', allow_duplicate=True),
    Input('main-plot','clickData'),
    State('user-session', 'data'),
    State('method-store', 'data'),
    State('dim-store',    'data'),
    prevent_initial_call=True,
)
def log_click(click_data, session, method, dims):
    """Registra clicks sobre puntos del scatter."""
    if not session or not click_data:
        return no_update
    uid   = session['user_id']
    start = session['session_start']
    pts   = click_data.get('points', [])
    if not pts:
        return no_update
    p = pts[0]
    x = p.get('x', '')
    y = p.get('y', '')
    z = p.get('z', '')
    txt = p.get('text', '')
    coord_x = f'{x:.4f}' if isinstance(x, float) else str(x)
    coord_y = f'{y:.4f}' if isinstance(y, float) else str(y)
    detail  = f'z={z:.4f} | {txt}' if z != '' else str(txt)
    _append_event(uid, start, 'click_point', method=method, dims=str(dims),
                  x_coord=coord_x, y_coord=coord_y, detail=detail)
    return no_update


@app.callback(
    Output('user-session','data', allow_duplicate=True),
    Input('main-plot','figure'),
    State('user-session', 'data'),
    State('method-store', 'data'),
    State('dim-store',    'data'),
    prevent_initial_call=True,
)
def log_legend_toggle(figure, session, method, dims):
    """Registra cambios de visibilidad de superclases (clicks en la leyenda)."""
    if not session or not figure:
        return no_update
    
    uid   = session['user_id']
    start = session['session_start']
    
    visibility_state = _get_superclass_visibility_state(figure)
    _append_event(uid, start, 'legend_toggle', method=method, dims=str(dims),
                  superclass_visibility=visibility_state, detail='Superclass visibility changed')
    
    return no_update



# ── Clientside: captura timestamps de eventos de mouse del plot ───────────────
clientside_callback(
    """
    function(relayoutData, session, method, dims) {
        if (!session) return window.dash_clientside.no_update;

        const mainPlot = document.getElementById('main-plot');
        if (!mainPlot || mainPlot._mouseLogAttached) return window.dash_clientside.no_update;
        mainPlot._mouseLogAttached = true;
        mainPlot._lastMousedown = null;

        // Captura duración de drag (pan / rotación sostenida)
        mainPlot.addEventListener('mousedown', function(e) {
            mainPlot._lastMousedown = {
                t: Date.now(),
                x: e.clientX,
                y: e.clientY,
                button: e.button  // 0=izquierdo, 1=medio, 2=derecho
            };
        });

        mainPlot.addEventListener('mouseup', function(e) {
            if (!mainPlot._lastMousedown) return;
            const duration_ms = Date.now() - mainPlot._lastMousedown.t;
            const dx = Math.abs(e.clientX - mainPlot._lastMousedown.x);
            const dy = Math.abs(e.clientY - mainPlot._lastMousedown.y);
            const isDrag = dx > 5 || dy > 5;
            const evtType = isDrag ? 'drag' : 'click_canvas';
            // Enviar al servidor vía store temporal (log-event-store)
            // No podemos escribir CSV aquí, se usa el store para relayar
            const logData = {
                event_type: evtType,
                duration_ms: duration_ms,
                button: mainPlot._lastMousedown.button,
                x: e.clientX,
                y: e.clientY,
                ts: new Date().toISOString()
            };
            // Guardar en atributo para que el callback de Python lo lea
            mainPlot._lastLogEvent = logData;
            mainPlot._lastMousedown = null;
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output('log-event-store', 'data'),
    Input('main-plot', 'relayoutData'),
    State('user-session', 'data'),
    State('method-store', 'data'),
    State('dim-store',    'data'),
    prevent_initial_call=True,
)


# ── Gizmo: anima los ejes según cámara (3D) o los fija (2D) ──────────────────
clientside_callback(
    """
    function(cameraData, dims) {
        const L = 32;

        function makeSVG(lines) {
            const stroke = 'stroke-linecap="round" stroke-width="2.5"';
            return `<svg viewBox="-50 -50 100 100" width="80" height="80" xmlns="http://www.w3.org/2000/svg">
                ${lines.map(l =>
                    `<line x1="0" y1="0" x2="${l.x2.toFixed(2)}" y2="${l.y2.toFixed(2)}"
                           stroke="${l.color}" ${stroke}/>
                     <circle cx="${l.x2.toFixed(2)}" cy="${l.y2.toFixed(2)}" r="4" fill="${l.color}"/>
                     <text x="${l.tx.toFixed(2)}" y="${l.ty.toFixed(2)}"
                           fill="${l.color}" font-size="12" font-family="monospace"
                           font-weight="bold">${l.label}</text>`
                ).join('')}
            </svg>`;
        }

        const el = document.getElementById('gizmo-container');
        if (!el) return window.dash_clientside.no_update;

        // ── 2D: ejes fijos ────────────────────────────────────────────────────
        if (dims !== 3) {
            el.innerHTML = makeSVG([
                {x2: L,  y2: 0,  tx: L+5, ty: 4,    color:'#e63946', label:'X'},
                {x2: 0,  y2:-L,  tx: 4,   ty:-L-2,  color:'#2196f3', label:'Y'},
            ]);
            return window.dash_clientside.no_update;
        }

        // ── 3D: proyectar ejes mundo usando eye de Plotly ────────────────────
        if (!cameraData || cameraData.type === 'cloud') {
            return window.dash_clientside.no_update;
        }

        const eye = cameraData.eye || {x:1.5, y:1.5, z:1.5};
        const ex = eye.x != null ? eye.x : 1.5;
        const ey = eye.y != null ? eye.y : 1.5;
        const ez = eye.z != null ? eye.z : 1.5;

        // Aplicar el mismo swap que el minimap Three.js:
        // Plotly(x, y, z) → Three(-x, z, y)  (negar X corrige handedness)
        const tx = -ex, ty = ez, tz = ey;

        // Dirección de visión en espacio Three (eye → origin)
        const len = Math.sqrt(tx*tx + ty*ty + tz*tz) || 1;
        const dx = -tx/len, dy = -ty/len, dz = -tz/len;

        // up world = (0,1,0) en Three.js (= eje Z de Plotly)
        // right = forward × up
        let rX = dy*1 - dz*0;
        let rY = dz*0 - dx*1;
        let rZ = dx*0 - dy*0;
        const rLen = Math.sqrt(rX*rX + rY*rY + rZ*rZ) || 1;
        rX/=rLen; rY/=rLen; rZ/=rLen;

        // real up = right × (-forward)
        let uX = rY*(-dz) - rZ*(-dy);
        let uY = rZ*(-dx) - rX*(-dz);
        let uZ = rX*(-dy) - rY*(-dx);
        const uLen = Math.sqrt(uX*uX + uY*uY + uZ*uZ) || 1;
        uX/=uLen; uY/=uLen; uZ/=uLen;

        // Proyectar cada eje de Plotly al espacio Three con el mismo swap
        // antes de hacer dot con right/up de la cámara
        function swapAxis(ax, ay, az) { return [-ax, az, ay]; }
        function project(ax, ay, az) {
            const [sx,sy,sz] = swapAxis(ax, ay, az);
            return [
                 (sx*rX + sy*rY + sz*rZ) * L,   // screen X
                -(sx*uX + sy*uY + sz*uZ) * L    // screen Y (SVG Y invertido)
            ];
        }

        const [xxS,xyS] = project(1,0,0);   // eje X Plotly → rojo
        const [yxS,yyS] = project(0,1,0);   // eje Y Plotly → azul
        const [zxS,zyS] = project(0,0,1);   // eje Z Plotly → verde
        const OFF = 7;

        el.innerHTML = makeSVG([
            {x2:xxS, y2:xyS, tx:xxS+Math.sign(xxS||1)*OFF, ty:xyS+Math.sign(xyS||1)*OFF, color:'#e63946', label:'X'},
            {x2:yxS, y2:yyS, tx:yxS+Math.sign(yxS||1)*OFF, ty:yyS+Math.sign(yyS||1)*OFF, color:'#2196f3', label:'Y'},
            {x2:zxS, y2:zyS, tx:zxS+Math.sign(zxS||1)*OFF, ty:zyS+Math.sign(zyS||1)*OFF, color:'#4caf50', label:'Z'},
        ]);

        return window.dash_clientside.no_update;
    }
    """,
    Output('gizmo-container', 'children'),
    Input('camera-store', 'data'),
    Input('dim-store',    'data'),
    prevent_initial_call=False,
)


if __name__ == '__main__':
    print('\n'+'='*50)
    print(f'Técnicas: {[ALL_METHODS[m] for m in available_methods]}')
    print(f'[LOG] Usuario #{_SESSION_UID:04d} | Sesión: {_SESSION_START}')
    print('http://127.0.0.1:8050')
    print('='*50+'\n')
    app.run(debug=False, port=8050)