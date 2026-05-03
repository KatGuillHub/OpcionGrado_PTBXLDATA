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

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, Patch, no_update, clientside_callback
import dash_bootstrap_components as dbc

# ── Rutas relativas (no requieren modificación) ────────────────────────────────
# Todos los archivos se buscan relativos a la ubicación de este script.
BASE     = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(BASE, 'res', 'projections_no_prereduction')
DB_CSV   = os.path.join(BASE, 'ptb-xl-data', 'ptbxl_database.csv')
SCP_CSV  = os.path.join(BASE, 'ptb-xl-data', 'scp_statements.csv')
LABEL_CSV= os.path.join(BASE, 'csv', 'ptbxl_label.csv')

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
def make_figure(method, dims, camera_state=None):
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
        # Reinyectar cámara guardada para evitar reset al cambiar herramienta
        # Si no hay cámara guardada, usar la posición default
        if camera_state and isinstance(camera_state, dict) and camera_state.get('type') == 'camera':
            camera = dict(
                eye=camera_state.get('eye', dict(x=1.5,y=1.5,z=1.5)),
                center=camera_state.get('center', dict(x=0,y=0,z=0)),
                projection=dict(type='orthographic')
            )
        else:
            camera = dict(eye=dict(x=1.5,y=1.5,z=1.5),
                         center=dict(x=0,y=0,z=0),
                         projection=dict(type='orthographic'))
        layout = go.Layout(
            title=dict(text=title,font=dict(size=14,color='#ecf0f1'),x=0.5),
            paper_bgcolor='#1a1a2e',
            # uirevision='constant' en layout Y en scene para máxima protección
            uirevision='3d_constant',
            scene=dict(
                xaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showbackground=False),
                yaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showbackground=False),
                zaxis=dict(showgrid=False,zeroline=False,showticklabels=False,showbackground=False),
                bgcolor='#1a1a2e',
                # aspectmode='manual' evita que scene.aspectratio dispare redraws
                aspectmode='manual',
                aspectratio=dict(x=1,y=1,z=1),
                uirevision='scene_constant',
                camera=camera),
            legend=LEGEND_STYLE,
            margin=dict(l=0,r=0,t=40,b=0), height=650)
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
            dcc.Graph(id='main-plot',figure=make_figure(default_method,2),
                      config={'scrollZoom':True,'displayModeBar':True,
                              'modeBarButtonsToRemove':['select2d','lasso2d']},
                      style={'borderRadius':'8px','overflow':'hidden'}),
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
            ], style=CARD_STYLE),
        ], width=3),
    ]),

    dcc.Store(id='camera-store'),
    dcc.Store(id='dim-store',    data=2),
    dcc.Store(id='method-store', data=default_method),

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
    State('camera-store',  'data'),
)
def update_on_change(method, dims, camera_state):
    # En 3D: reinyectar la cámara guardada para que no se resetee al cambiar técnica
    main_fig = make_figure(method, dims, camera_state if dims == 3 else None)
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
    if not relayout_data or dims != 3:
        return no_update

    # Ignorar eventos sin datos de cámara (solo dragmode/autosize)
    keys = set(relayout_data.keys())
    if keys <= {'scene.dragmode', 'dragmode', 'autosize', 'scene.aspectratio',
                'scene.aspectmode'}:
        return no_update

    cam = relayout_data.get('scene.camera')
    if cam:
        eye    = cam.get('eye',    {'x':1.5,'y':1.5,'z':1.5})
        center = cam.get('center', {'x':0,'y':0,'z':0})
        zoom   = (cam.get('projection') or {}).get('scale', 1.0) or 1.0
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


if __name__ == '__main__':
    print('\n'+'='*50)
    print(f'Técnicas: {[ALL_METHODS[m] for m in available_methods]}')
    print('http://127.0.0.1:8050')
    print('='*50+'\n')
    app.run(debug=False, port=8050)