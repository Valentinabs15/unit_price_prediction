import base64
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción de Precio Unitario", page_icon="💎", layout="centered")

def set_background():
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
                    .stApp:before {{
                        content: "";
                        position: fixed;
                        inset: 0;
                        background: rgba(0,0,0,.58);
                        z-index: 0;
                    }}
                    .block-container {{ position: relative; z-index: 1; }}
                    </style>
                """, unsafe_allow_html=True)
                return
    
st.markdown("""
<style>
/* ===== Fondo y overlay siguen igual (definido antes) ===== */

/* ===== Card visual ===== */
.hero {
  background: rgba(20,20,20,0.60);
  -webkit-backdrop-filter: blur(6px);
  backdrop-filter: blur(6px);
  border-radius: 18px;
  padding: 28px 26px;
  box-shadow: 0 14px 40px rgba(0,0,0,.35);
  border: 1px solid rgba(255,255,255,0.22);
}

/* ===== Texto global blanco con sombra (NO inputs) ===== */
.hero * { 
  color: #ffffff !important;
  text-shadow: 0 1px 2px rgba(0,0,0,.85);
}
/* ===== Inputs blancos con texto negro ===== */
.hero input, .hero textarea { 
  color: #111 !important; background: #ffffff !important; text-shadow: none !important;
}
.hero input::placeholder, .hero textarea::placeholder { color: #6b7280 !important; }
.hero [data-baseweb="select"] * { color: #111 !important; text-shadow: none !important; }
.hero .stNumberInput input, .hero .stTextInput input, .hero .stSelectbox > div > div { color: #111 !important; text-shadow: none !important; }

/* ===== BOTÓN ===== */
.hero .stButton>button { width: 100%; font-weight: 700; box-shadow: 0 6px 16px rgba(0,0,0,.25); }

/* ===== Chip de clúster ===== */
.chip {
  display: inline-block; padding: 8px 12px; border-radius: 999px;
  background: rgba(0,0,0,0.55);
  border: 1px solid rgba(255,255,255,0.35);
  box-shadow: 0 4px 12px rgba(0,0,0,.25);
  margin: 6px 0 14px 0; font-weight: 800; color: #fff !important;
}

/* ===== Badge utilitario para títulos como "Período 1" ===== */
.badge {
  display: inline-block; padding: 6px 10px; border-radius: 8px;
  background: rgba(0,0,0,0.65); color: #fff; font-weight: 800; margin: 6px 0 6px;
}

/* ===== Forzar etiquetas de widgets a BLANCO + "subfondo" oscuro ===== */
.stApp [data-testid="stWidgetLabel"], 
.stApp [data-testid="stWidgetLabel"] *, 
.stApp label {
  color: #fff !important; opacity: 1 !important; text-shadow: 0 1px 2px rgba(0,0,0,.85) !important;
}
/* Fondo detrás del texto del label */
.stApp [data-testid="stWidgetLabel"] p, 
.stApp [data-testid="stWidgetLabel"] span, 
.stApp [data-testid="stWidgetLabel"] label {
  display: inline-block; padding: 4px 8px; border-radius: 6px;
  background: rgba(0,0,0,0.55);
}
</style>
""", unsafe_allow_html=True)
set_background()




st.markdown("""
<style>
/* ===== Fondo y overlay siguen igual (definido antes) ===== */

/* ===== Card visual ===== */
.hero {
  background: rgba(20,20,20,0.60);
  -webkit-backdrop-filter: blur(6px);
  backdrop-filter: blur(6px);
  border-radius: 18px;
  padding: 28px 26px;
  box-shadow: 0 14px 40px rgba(0,0,0,.35);
  border: 1px solid rgba(255,255,255,0.22);
}

/* ===== Texto global blanco con sombra (NO inputs) ===== */
.hero * { 
  color: #ffffff !important;
  text-shadow: 0 1px 2px rgba(0,0,0,.85);
}
/* ===== Inputs blancos con texto negro ===== */
.hero input, .hero textarea { 
  color: #111 !important; background: #ffffff !important; text-shadow: none !important;
}
.hero input::placeholder, .hero textarea::placeholder { color: #6b7280 !important; }
.hero [data-baseweb="select"] * { color: #111 !important; text-shadow: none !important; }
.hero .stNumberInput input, .hero .stTextInput input, .hero .stSelectbox > div > div { color: #111 !important; text-shadow: none !important; }

/* ===== BOTÓN ===== */
.hero .stButton>button { width: 100%; font-weight: 700; box-shadow: 0 6px 16px rgba(0,0,0,.25); }

/* ===== Chip de clúster ===== */
.chip {
  display: inline-block; padding: 8px 12px; border-radius: 999px;
  background: rgba(0,0,0,0.55);
  border: 1px solid rgba(255,255,255,0.35);
  box-shadow: 0 4px 12px rgba(0,0,0,.25);
  margin: 6px 0 14px 0; font-weight: 800; color: #fff !important;
}

/* ===== Badge utilitario para títulos como "Período 1" ===== */
.badge {
  display: inline-block; padding: 6px 10px; border-radius: 8px;
  background: rgba(0,0,0,0.65); color: #fff; font-weight: 800; margin: 6px 0 6px;
}

/* ===== Forzar etiquetas de widgets a BLANCO + "subfondo" oscuro ===== */
.stApp [data-testid="stWidgetLabel"], 
.stApp [data-testid="stWidgetLabel"] *, 
.stApp label {
  color: #fff !important; opacity: 1 !important; text-shadow: 0 1px 2px rgba(0,0,0,.85) !important;
}
/* Fondo detrás del texto del label */
.stApp [data-testid="stWidgetLabel"] p, 
.stApp [data-testid="stWidgetLabel"] span, 
.stApp [data-testid="stWidgetLabel"] label {
  display: inline-block; padding: 4px 8px; border-radius: 6px;
  background: rgba(0,0,0,0.55);
}
</style>
""", unsafe_allow_html=True)
# ===== Constantes
ZONAS = ["Centro", "Norte Chico", "Norte Grande", "Sur"]
ZONAS_MAP = {"Centro": 0, "Norte Chico": 1, "Norte Grande": 2, "Sur": 3}
NOMBRES_CLUSTER = {0:"Gran Tronadura",1:"Tronadura Fuerte",2:"Tronadura Intermedia",3:"Tronadura Estándar"}
CAMPOS_MACRO = ["CLP/USD","IPC CL (base 2018)","FERT AMM (Caribe US Spot)","FERTECON AGAN BlackSea","CPI USA","ENAP Diesel"]
CAMPOS_MODELO_DF = ["Volume","Zone","CLP_USD_FX_MONTHLY","IPC_BASE2018_CP_CL_MONTHLY","FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY","FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP","CPI_USA_MONTHLY","FUENTE_ENAP_CHL_DL_MONTHLY"]

BASE = Path(__file__).parent

@st.cache_resource(show_spinner=False)
def cargar_modelos_desde_raiz():
    modelos = {i: joblib.load(BASE / f"modelo_cluster_{i}_General.pkl") for i in range(4)}
    kmeans = joblib.load(BASE / "kmeans.pkl") if (BASE / "kmeans.pkl").exists() else None
    scaler_cluster = joblib.load(BASE / "scaler_cluster.pkl") if (BASE / "scaler_cluster.pkl").exists() else None
    return modelos, kmeans, scaler_cluster

def _cluster_candidates(volume, zid):
    return [np.array([[volume, zid]], dtype=float), np.array([[volume, zid, 0.0]], dtype=float)]

def _predict_candidates(volume, zid, valores):
    df_full = pd.DataFrame([[volume, zid]+list(valores)], columns=CAMPOS_MODELO_DF)
    df_no_zone = pd.DataFrame([[volume]+list(valores)], columns=["Volume"]+CAMPOS_MODELO_DF[2:])
    arr_full = np.array([volume, zid]+list(valores), dtype=float).reshape(1,-1)
    arr_no_zone = np.array([volume]+list(valores), dtype=float).reshape(1,-1)
    return [df_full, df_no_zone, arr_full, arr_no_zone]

def try_detect_cluster(volume, zona, kmeans, scaler_cluster):
    if (kmeans is None) or (scaler_cluster is None):
        return None
    zid = ZONAS_MAP[zona]
    last = None
    for Xc in _cluster_candidates(volume, zid):
        try:
            scaled = scaler_cluster.transform(Xc)
            cid = int(kmeans.predict(scaled)[0])
            return cid
        except Exception as e:
            last = e
            continue
    return None

def predecir_varios_periodos(volume, zona, lista_valores, modelos, kmeans=None, scaler_cluster=None, cluster_manual=None):
    zid = ZONAS_MAP[zona]
    if (kmeans is not None) and (scaler_cluster is not None):
        cid = try_detect_cluster(volume, zona, kmeans, scaler_cluster)
        if cid is None:
            raise RuntimeError("No se pudo transformar/predicir clúster con el scaler/kmeans actuales.")
        cluster = cid
    else:
        if cluster_manual is None:
            raise RuntimeError("Faltan kmeans.pkl o scaler_cluster.pkl. Sube ambos para clasificar automáticamente.")
        cluster = int(cluster_manual)

    modelo = modelos.get(cluster)
    if modelo is None:
        raise RuntimeError(f"Sin modelo para clúster {cluster}")

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

# ===== Carga modelos
try:
    modelos, kmeans, scaler_cluster = cargar_modelos_desde_raiz()
    modelos_ok = True
except Exception as e:
    modelos_ok = False
    st.error("No se pudieron cargar los modelos desde la raíz del repositorio.")
    st.code(str(e))

# ===== UI
with st.container():
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("<h1>💎 Predicción de Precio Unitario</h1>", unsafe_allow_html=True)
    st.markdown('<p class="sub">Simula escenarios por zona y volumen con variables macroeconómicas</p>', unsafe_allow_html=True)

    volume = st.number_input("📦 Volumen (toneladas)", min_value=0.0, value=200.0, step=1.0)
    zona = st.selectbox("📍 Zona", ZONAS, index=0)
    periodos = st.number_input("🗓️ N° de períodos", min_value=1, max_value=12, value=1, step=1)

    # Chip de clúster detectado (preview)
    if modelos_ok and (kmeans is not None and scaler_cluster is not None):
        cid = try_detect_cluster(volume, zona, kmeans, scaler_cluster)
        if cid is not None:
            cname = NOMBRES_CLUSTER.get(cid, f"Clúster {cid}")
            st.markdown(f'<div class="chip">Clúster detectado: <b>{cid}</b> — {cname}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chip">Clúster: no detectable con el scaler/kmeans actuales</div>', unsafe_allow_html=True)
    else:
        st.warning("Para **clasificar automáticamente** sube **kmeans.pkl** y **scaler_cluster.pkl** a la raíz del repositorio.")

    lista_valores = []
    for i in range(int(periodos)):
        st.markdown(f'<span class="badge">Período {i+1}</span>', unsafe_allow_html=True)
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
                    volume, zona, lista_valores, modelos, kmeans=kmeans, scaler_cluster=scaler_cluster, cluster_manual=None
                )
                st.success(f"📌 Clúster asignado: {cid} — {cname}")
                for i, p in enumerate(preds):
                    st.write(f"📊 Período {i+1}: **${p:,.4f}**")
                st.balloons()
            except Exception as e:
                st.error("Ocurrió un error durante la predicción.")
                st.code(str(e))

    st.markdown('</div>', unsafe_allow_html=True)
