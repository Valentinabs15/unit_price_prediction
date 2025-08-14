import base64
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================
# Setup y estilo (hero)
# ============================
st.set_page_config(page_title="Predicción de Precio Unitario", page_icon="💎", layout="centered")

def set_background():
    # Busca fondo en raíz (fondo.*) o en assets/
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    for base in [Path(__file__).parent, Path(__file__).parent / "assets"]:
        for ext in exts:
            p = base / f"fondo{ext}"
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode()
                st.markdown(f"""
                    <style>
                    .stApp {{
                        background-image: url("data:image/{p.suffix[1:]};base64,{b64}");
                        background-size: cover;
                        background-attachment: fixed;
                        background-position: center;
                    }}
                    </style>
                """, unsafe_allow_html=True)
                return
    st.markdown("""
        <style>
        .stApp {{
            background: linear-gradient(180deg, rgba(245,246,250,1) 0%, rgba(255,255,255,1) 60%);
        }}
        </style>
    """, unsafe_allow_html=True)

set_background()

st.markdown("""
<style>
.hero {
  background: rgba(255,255,255,0.72);
  -webkit-backdrop-filter: blur(6px);
  backdrop-filter: blur(6px);
  border-radius: 18px;
  padding: 28px 26px;
  box-shadow: 0 10px 30px rgba(0,0,0,.12);
  border: 1px solid rgba(255,255,255,0.6);
}
.hero h1 { font-size: 44px; line-height: 1.1; margin: 4px 0 10px 0; }
.hero p.sub { color: #3a3a3a; margin-bottom: 18px; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# ============================
# Constantes del modelo (zonas del notebook)
# ============================
ZONAS = ["Centro", "Norte Chico", "Norte Grande", "Sur"]
ZONAS_MAP = {"Centro": 0, "Norte Chico": 1, "Norte Grande": 2, "Sur": 3}

NOMBRES_CLUSTER = {
    0: "Gran Tronadura",
    1: "Tronadura Fuerte",
    2: "Tronadura Intermedia",
    3: "Tronadura Estándar",
}

CAMPOS_MACRO = [
    "CLP/USD",
    "IPC CL (base 2018)",
    "FERT AMM (Caribe US Spot)",
    "FERTECON AGAN BlackSea",
    "CPI USA",
    "ENAP Diesel",
]

CAMPOS_MODELO_DF = [
    "Volume","Zone",
    "CLP_USD_FX_MONTHLY","IPC_BASE2018_CP_CL_MONTHLY",
    "FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY","FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP",
    "CPI_USA_MONTHLY","FUENTE_ENAP_CHL_DL_MONTHLY",
]

# ============================
# Carga de artefactos desde la raíz del repo (como en tu GitHub)
# ============================
BASE = Path(__file__).parent

@st.cache_resource(show_spinner=False)
def cargar_modelos_desde_raiz():
    # Modelos por clúster (obligatorios)
    modelos_paths = {i: BASE / f"modelo_cluster_{i}_General.pkl" for i in range(4)}
    faltan = [p.name for p in modelos_paths.values() if not p.exists()]
    if faltan:
        raise FileNotFoundError("Faltan archivos de modelo en la raíz del repo: " + ", ".join(faltan))

    modelos = {i: joblib.load(modelos_paths[i]) for i in range(4)}

    # Clusterización automática (opcional): kmeans + scaler_cluster
    kmeans_path = BASE / "kmeans.pkl"
    scaler_cluster_path = BASE / "scaler_cluster.pkl"
    tiene_auto_cluster = kmeans_path.exists() and scaler_cluster_path.exists()
    kmeans = joblib.load(kmeans_path) if tiene_auto_cluster else None
    scaler_cluster = joblib.load(scaler_cluster_path) if tiene_auto_cluster else None

    return modelos, kmeans, scaler_cluster

def _cluster_candidates(volume, zid):
    return [
        np.array([[volume, zid]], dtype=float),
        np.array([[volume, zid, 0.0]], dtype=float),  # por si el scaler espera 3 features
    ]

def _predict_candidates(volume, zid, valores):
    df_full = pd.DataFrame([[volume, zid]+list(valores)], columns=CAMPOS_MODELO_DF)
    df_no_zone = pd.DataFrame([[volume]+list(valores)], columns=["Volume"]+CAMPOS_MODELO_DF[2:])
    arr_full = np.array([volume, zid]+list(valores), dtype=float).reshape(1,-1)
    arr_no_zone = np.array([volume]+list(valores), dtype=float).reshape(1,-1)
    return [df_full, df_no_zone, arr_full, arr_no_zone]

def predecir_varios_periodos(volume, zona, lista_valores, modelos, kmeans=None, scaler_cluster=None, cluster_manual=None):
    zid = ZONAS_MAP[zona]

    # 1) Resolver clúster
    if (kmeans is not None) and (scaler_cluster is not None):
        scaled, last = None, None
        for Xc in _cluster_candidates(volume, zid):
            try: scaled = scaler_cluster.transform(Xc); break
            except Exception as e: last = e
        if scaled is None:
            raise RuntimeError(f"No se pudo transformar para cluster: {last}")
        cluster = int(kmeans.predict(scaled)[0])
    else:
        if cluster_manual is None:
            raise RuntimeError("Falta kmeans.pkl o scaler_cluster.pkl en la raíz; selecciona el clúster manualmente.")
        cluster = int(cluster_manual)

    modelo = modelos.get(cluster)
    if modelo is None:
        raise RuntimeError(f"Sin modelo para clúster {cluster}")

    # 2) Predicción por período
    preds = []
    for vals in lista_valores:
        y, err = None, None
        for X in _predict_candidates(volume, zid, vals):
            try:
                y = float(modelo.predict(X)[0]); break
            except Exception as e:
                err = e
        if y is None:
            raise RuntimeError(f"No se pudo predecir (clúster {cluster}): {err}")
        preds.append(y)

    return cluster, NOMBRES_CLUSTER.get(cluster, f"Clúster {cluster}"), preds

# ============================
# Carga modelos
# ============================
try:
    modelos, kmeans, scaler_cluster = cargar_modelos_desde_raiz()
    modelos_ok = True
except Exception as e:
    modelos_ok = False
    st.error("No se pudieron cargar los modelos desde la raíz del repositorio.")
    st.code(str(e))

# ============================
# UI (Hero)
# ============================
with st.container():
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("<h1>💎 Predicción de Precio Unitario</h1>", unsafe_allow_html=True)
    st.markdown('<p class="sub">Simula escenarios por zona y volumen con variables macroeconómicas</p>', unsafe_allow_html=True)

    volume = st.number_input("📦 Volumen (toneladas)", min_value=0.0, value=200.0, step=1.0)
    zona = st.selectbox("📍 Zona", ZONAS, index=0)
    periodos = st.number_input("🗓️ N° de períodos", min_value=1, max_value=12, value=1, step=1)

    # Si NO hay kmeans+scaler_cluster, permite elegir clúster manualmente
    cluster_manual = None
    if modelos_ok and (kmeans is None or scaler_cluster is None):
        st.info("No se encontró **kmeans.pkl** o **scaler_cluster.pkl** en la raíz. Selecciona el **modelo de clúster** manualmente.")
        cluster_manual = st.selectbox("Modelo por clúster", options=[0,1,2,3],
                                      format_func=lambda i: f"{i} — {NOMBRES_CLUSTER.get(i, 'Clúster ' + str(i))}")

    lista_valores = []
    for i in range(int(periodos)):
        st.markdown(f"**Período {i+1}**")
        cols = st.columns(3)
        vals = []
        for j, campo in enumerate(CAMPOS_MACRO):
            with cols[j % 3]:
                val = st.number_input(f"{campo} (P{i+1})", value=0.0, key=f"{campo}_{i}")
                vals.append(val)
        lista_valores.append(vals)
        if i < periodos-1:
            st.divider()

    if st.button("🔮 Predecir", use_container_width=True):
        if not modelos_ok:
            st.error("Modelos no disponibles todavía.")
        else:
            try:
                cid, cname, preds = predecir_varios_periodos(
                    volume, zona, lista_valores, modelos, kmeans=kmeans, scaler_cluster=scaler_cluster, cluster_manual=cluster_manual
                )
                st.success(f"📌 Clúster asignado: {cid} — {cname}")
                for i, p in enumerate(preds):
                    st.write(f"📊 Período {i+1}: **${p:,.4f}**")
                st.balloons()
            except Exception as e:
                st.error("Ocurrió un error durante la predicción.")
                st.code(str(e))

    st.markdown('</div>', unsafe_allow_html=True)
