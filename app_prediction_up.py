import base64
import io
import re
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =====================================================
# Apariencia (fondo: busca ./assets/fondo.* o gradiente)
# =====================================================
def set_background_auto():
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    bases = [Path(__file__).parent / "assets", Path(__file__).parent]
    for base in bases:
        for ext in exts:
            p = base / f"fondo{ext}"
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode()
                st.markdown(
                    f"""
                    <style>
                    .stApp {{
                        background-image: url("data:image/{p.suffix[1:]};base64,{b64}");
                        background-size: cover;
                        background-attachment: fixed;
                        background-position: center;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                return
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, rgba(245,246,250,1) 0%, rgba(255,255,255,1) 60%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background_auto()

st.set_page_config(page_title="Predicción de Precio Unitario", page_icon="💥", layout="wide")
st.title("💥 Predicción de Precio Unitario")
st.caption("Versión para Streamlit Cloud. Descarga de modelos por ZIP o URLs y zonas alineadas al notebook.")

# =============================
# Zonas fijas según notebook
# =============================
ZONAS = ["Centro", "Norte Chico", "Norte Grande", "Sur"]
ZONAS_MAP = {"Centro": 0, "Norte Chico": 1, "Norte Grande": 2, "Sur": 3}

# Etiquetas legibles para clúster
NOMBRES_CLUSTER = {
    0: "Gran Tronadura",
    1: "Tronadura Fuerte",
    2: "Tronadura Intermedia",
    3: "Tronadura Estándar",
}

# Campos macro visibles (el pipeline del modelo maneja su propio escalado)
CAMPOS_MACRO = [
    "CLP/USD",
    "IPC CL (base 2018)",
    "FERT AMM (Caribe US Spot)",
    "FERTECON AGAN BlackSea",
    "CPI USA",
    "ENAP Diesel",
]

# Columnas con nombres de entrenamiento (por si el modelo espera DataFrame)
CAMPOS_MODELO_DF = [
    "Volume",
    "Zone",
    "CLP_USD_FX_MONTHLY",
    "IPC_BASE2018_CP_CL_MONTHLY",
    "FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY",
    "FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP",
    "CPI_USA_MONTHLY",
    "FUENTE_ENAP_CHL_DL_MONTHLY",
]

# =====================================================
# Helpers para descarga (GitHub/Drive) y ZIP
# =====================================================
def _github_to_raw(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/")
    return url

def _drive_direct(url: str) -> str:
    if "uc?export=download&id=" in url or "uc?id=" in url:
        return url.replace("uc?id=", "uc?export=download&id=")
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    return url

def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if "github.com" in url:
        return _github_to_raw(url)
    if "drive.google.com" in url:
        return _drive_direct(url)
    return url

def download_file(url: str, dest: Path) -> None:
    url = _normalize_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)

EXPECTED_FILENAMES = {
    "scaler.pkl",
    "scaler_cluster.pkl",
    "kmeans.pkl",
    "modelo_cluster_0_General.pkl",
    "modelo_cluster_1_General.pkl",
    "modelo_cluster_2_General.pkl",
    "modelo_cluster_3_General.pkl",
}

def download_and_extract_zip(zip_url: str, dest_dir: Path) -> list:
    url = _normalize_url(zip_url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    content = io.BytesIO(r.content)
    moved = []
    with zipfile.ZipFile(content) as z:
        tmp = dest_dir / "__tmp_zip_extract__"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        z.extractall(tmp)
        for p in tmp.rglob("*"):
            if p.is_file() and p.name in EXPECTED_FILENAMES:
                target = dest_dir / p.name
                target.write_bytes(p.read_bytes())
                moved.append(str(target))
        shutil.rmtree(tmp, ignore_errors=True)
    return moved

# =====================================================
# Carga de artefactos
# =====================================================
@st.cache_resource(show_spinner=False)
def cargar_modelos(ruta: str):
    base = Path(ruta)
    # scaler puede ser scaler.pkl o scaler_cluster.pkl
    scaler_path = None
    for name in ["scaler.pkl", "scaler_cluster.pkl"]:
        p = base / name
        if p.exists():
            scaler_path = p
            break
    if scaler_path is None:
        raise FileNotFoundError("Falta scaler.pkl o scaler_cluster.pkl en " + str(base))

    kmeans_path = base / "kmeans.pkl"
    if not kmeans_path.exists():
        raise FileNotFoundError("Falta kmeans.pkl en " + str(base))

    modelos_paths = {i: base / f"modelo_cluster_{i}_General.pkl" for i in range(4)}
    faltan = [str(p) for p in modelos_paths.values() if not p.exists()]
    if faltan:
        raise FileNotFoundError("Faltan modelos por clúster:\n- " + "\n- ".join(faltan))

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    modelos = {i: joblib.load(modelos_paths[i]) for i in range(4)}
    return scaler, kmeans, modelos, scaler_path.name

def _cluster_transform_candidates(volume, zona_id):
    return [
        np.array([[volume, zona_id]], dtype=float),
        np.array([[volume, zona_id, 0.0]], dtype=float),
    ]

def _prediction_candidates(volume, zona_id, valores):
    df_full = pd.DataFrame([[volume, zona_id] + list(valores)], columns=CAMPOS_MODELO_DF)
    df_no_zone = pd.DataFrame([[volume] + list(valores)], columns=["Volume"] + CAMPOS_MODELO_DF[2:])
    arr_full = np.array([volume, zona_id] + list(valores), dtype=float).reshape(1, -1)
    arr_no_zone = np.array([volume] + list(valores), dtype=float).reshape(1, -1)
    return [df_full, df_no_zone, arr_full, arr_no_zone]

def predecir_varios_periodos(volume, zona_str, lista_valores, scaler, kmeans, modelos):
    zona_id = ZONAS_MAP[zona_str]

    cluster_scaled = None
    last_exc = None
    for cand in _cluster_transform_candidates(volume, zona_id):
        try:
            cluster_scaled = scaler.transform(cand)
            break
        except Exception as e:
            last_exc = e
    if cluster_scaled is None:
        raise RuntimeError(f"No fue posible transformar las features para clustering: {last_exc}")

    cluster = int(kmeans.predict(cluster_scaled)[0])
    modelo = modelos.get(cluster)
    if modelo is None:
        raise RuntimeError(f"No hay modelo para el clúster {cluster}")

    preds = []
    last = None
    for vals in lista_valores:
        y = None
        for X in _prediction_candidates(volume, zona_id, vals):
            try:
                y = float(modelo.predict(X)[0])
                break
            except Exception as e:
                last = e
                continue
        if y is None:
            raise RuntimeError(f"No se pudo predecir con el modelo del clúster {cluster}: {last}")
        preds.append(y)

    return cluster, NOMBRES_CLUSTER.get(cluster, f"Clúster {cluster}"), preds

# =====================================================
# Sidebar: carpeta y descargas
# =====================================================
st.sidebar.header("⚙️ Configuración")
ruta_modelos = st.sidebar.text_input("Carpeta de modelos (.pkl)", value="modelos")

with st.sidebar.expander("⬇️ Descargar modelos desde URLs (GitHub / Drive)", expanded=False):
    st.caption("Pega las URLs de tus artefactos. Se guardarán en la carpeta indicada arriba.")
    colA, colB = st.columns(2)
    with colA:
        url_scaler = st.text_input("URL scaler (scaler.pkl o scaler_cluster.pkl)", value="")
        url_kmeans = st.text_input("URL kmeans.pkl", value="")
        url_m0 = st.text_input("URL modelo_cluster_0_General.pkl", value="")
    with colB:
        url_m1 = st.text_input("URL modelo_cluster_1_General.pkl", value="")
        url_m2 = st.text_input("URL modelo_cluster_2_General.pkl", value="")
        url_m3 = st.text_input("URL modelo_cluster_3_General.pkl", value="")
    if st.button("Descargar archivos", use_container_width=True):
        try:
            base = Path(ruta_modelos)
            if url_scaler:
                scaler_name = "scaler_cluster.pkl" if "scaler_cluster" in url_scaler else "scaler.pkl"
                download_file(url_scaler, base / scaler_name)
            if url_kmeans:
                download_file(url_kmeans, base / "kmeans.pkl")
            if url_m0:
                download_file(url_m0, base / "modelo_cluster_0_General.pkl")
            if url_m1:
                download_file(url_m1, base / "modelo_cluster_1_General.pkl")
            if url_m2:
                download_file(url_m2, base / "modelo_cluster_2_General.pkl")
            if url_m3:
                download_file(url_m3, base / "modelo_cluster_3_General.pkl")
            st.success("Descarga completada. Ve a 'Cargar modelos' o presiona Rerun.")
            st.cache_resource.clear()
        except Exception as e:
            st.error("Falló la descarga de uno o más archivos.")
            st.code(str(e))

with st.sidebar.expander("📦 Descargar un ZIP con TODOS los modelos", expanded=False):
    st.caption("Pega un único enlace a un ZIP (GitHub Release / Google Drive).")
    zip_url = st.text_input("URL del ZIP", value="", placeholder="https://drive.google.com/file/d/<ID>/view")
    if st.button("Descargar y extraer ZIP", use_container_width=True):
        try:
            moved = download_and_extract_zip(zip_url, Path(ruta_modelos))
            if moved:
                st.success("ZIP extraído. Archivos copiados:")
                for m in moved:
                    st.write("•", m)
                st.cache_resource.clear()
            else:
                st.warning("No se encontraron archivos esperados dentro del ZIP.")
        except Exception as e:
            st.error("No se pudo descargar o extraer el ZIP.")
            st.code(str(e))

# =====================================================
# Carga de modelos (fuera de try/except anterior)
# =====================================================
st.sidebar.markdown("---")
if st.sidebar.button("Cargar modelos ahora", use_container_width=True):
    st.experimental_rerun()

try:
    scaler, kmeans, modelos, scaler_name = cargar_modelos(ruta_modelos)
    st.success(f"Modelos cargados desde '{ruta_modelos}' (scaler: {scaler_name}).")
except Exception as e:
    st.error("No se pudieron cargar los modelos.")
    st.code(str(e))
    st.stop()

# =====================================================
# UI principal
# =====================================================
col1, col2 = st.columns(2)
with col1:
    volume = st.number_input("📦 Volumen (toneladas)", min_value=0.0, value=200.0, step=1.0)
with col2:
    zona = st.selectbox("📍 Zona", ZONAS)

periodos = st.number_input("🗓️ N° de períodos", min_value=1, max_value=12, value=1, step=1)

lista_valores = []
for i in range(int(periodos)):
    st.markdown(f"#### 🔁 Período {i+1}")
    cols = st.columns(3)
    valores = []
    for j, campo in enumerate(CAMPOS_MACRO):
        with cols[j % 3]:
            val = st.number_input(f"{campo} (P{i+1})", value=0.0, key=f"{campo}_{i}")
            valores.append(val)
    lista_valores.append(valores)

if st.button("Predecir", use_container_width=True):
    try:
        cluster_id, cluster_nombre, predicciones = predecir_varios_periodos(
            volume, zona, lista_valores, scaler, kmeans, modelos
        )
        st.success(f"📌 Clúster asignado: {cluster_id} - {cluster_nombre}")
        for i, pred in enumerate(predicciones):
            st.write(f"📊 Período {i+1}: **${pred:,.4f}**")
        st.balloons()
    except Exception as e:
        st.error("Ocurrió un error realizando la predicción.")
        st.code(str(e))
