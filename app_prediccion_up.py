import base64
import io
import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# =============================
# Descarga de modelos (GitHub / Google Drive)
# =============================
import re, requests, shutil

def _github_to_raw(url: str) -> str:
    # Convierte https://github.com/user/repo/blob/main/file.pkl -> raw.githubusercontent.com/user/repo/main/file.pkl
    if "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/")
    return url

def _drive_direct(url: str) -> str:
    """
    Normaliza links de Google Drive a forma directa uc?export=download&id=FILEID
    Acepta:
      - https://drive.google.com/file/d/<ID>/view?usp=sharing
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>
      - https://drive.google.com/uc?export=download&id=<ID>
    """
    # Si ya es uc con id, lo dejamos
    if "uc?export=download&id=" in url or "uc?id=" in url:
        return url.replace("uc?id=", "uc?export=download&id=")
    # Extraer ID del patrón /file/d/<ID>/
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        # Intento id=
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    return url  # fallback

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
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)


# =============================
# Apariencia: fondo
# =============================
def _encode_image(path: Path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background_auto():
    """
    Intenta cargar un archivo llamado 'fondo' con extensiones comunes
    desde ./assets o desde la carpeta del app. Si no existe, aplica
    un gradiente suave por defecto.
    """
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    bases = [Path(__file__).parent / "assets", Path(__file__).parent]
    for base in bases:
        for ext in exts:
            p = base / f"fondo{ext}"
            if p.exists():
                encoded = _encode_image(p)
                st.markdown(
                    f"""
                    <style>
                    .stApp {{
                        background-image: url("data:image/{p.suffix[1:]};base64,{encoded}");
                        background-size: cover;
                        background-attachment: fixed;
                        background-position: center;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                return
    # Fondo por defecto (gradiente)
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, rgba(245,246,250,1) 0%, rgba(255,255,255,1) 60%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_auto()

# =============================
# Configuración
# =============================
st.title("🔍 Predicción de Precio Unitario")
st.caption("Replica de la app original, ajustada para el notebook nuevo.")

# Rutas (como en la app antigua, por defecto el directorio actual)
ruta_modelos = st.text_input("🗂️ Carpeta de modelos (.pkl)", value="modelos")

# Zonas según el notebook (mapeo fijo)
ZONAS = ["Centro", "Norte Chico", "Norte Grande", "Sur"]
ZONAS_MAP = {"Centro": 0, "Norte Chico": 1, "Norte Grande": 2, "Sur": 3}

# Etiquetas legibles para clúster (display)
NOMBRES_CLUSTER = {
    0: "Gran Tronadura",
    1: "Tronadura Fuerte",
    2: "Tronadura Intermedia",
    3: "Tronadura Estándar"
}

# Campos macro (solo para UI). El pipeline del modelo se encarga del escalado.
CAMPOS_MACRO = [
    "CLP/USD",                      # CLP_USD_FX_MONTHLY
    "IPC CL (base 2018)",           # IPC_BASE2018_CP_CL_MONTHLY
    "FERT AMM (Caribe US Spot)",    # FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY
    "FERTECON AGAN BlackSea",       # FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP
    "CPI USA",                      # CPI_USA_MONTHLY
    "ENAP Diesel"                   # FUENTE_ENAP_CHL_DL_MONTHLY
]

# Columnas con nombres del notebook para intentar DataFrame si el modelo lo requiere
CAMPOS_MODELO_DF = [
    "Volume", "Zone",
    "CLP_USD_FX_MONTHLY", "IPC_BASE2018_CP_CL_MONTHLY",
    "FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY", "FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP",
    "CPI_USA_MONTHLY", "FUENTE_ENAP_CHL_DL_MONTHLY"
]

# =============================
# Carga de artefactos
# =============================
@st.cache_resource(show_spinner=False)
def cargar_modelos(ruta: str):
    """Carga scaler, kmeans y los 4 modelos por clúster.
    Es compatible con nombres antiguos (scaler.pkl) o nuevos (scaler_cluster.pkl).
    """
    base = Path(ruta)
    # scaler puede llamarse scaler.pkl o scaler_cluster.pkl
    scaler_path = None
    for name in ["scaler.pkl", "scaler_cluster.pkl"]:
        p = base / name
        if p.exists():
            scaler_path = p
            break
    if scaler_path is None:
        raise FileNotFoundError("No se encontró scaler.pkl ni scaler_cluster.pkl en " + str(base))

    kmeans_path = base / "kmeans.pkl"
    if not kmeans_path.exists():
        raise FileNotFoundError("No se encontró kmeans.pkl en " + str(base))

    modelos_paths = {i: base / f"modelo_cluster_{i}_General.pkl" for i in range(4)}
    faltan = [str(p) for p in modelos_paths.values() if not p.exists()]
    if faltan:
        raise FileNotFoundError("Faltan modelos por clúster:\n- " + "\n- ".join(faltan))

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    modelos = {i: joblib.load(modelos_paths[i]) for i in range(4)}
    return scaler, kmeans, modelos, str(scaler_path.name)

def _cluster_transform_candidates(volume, zona_id):
    """Devuelve candidatos de forma para el scaler del clúster (2 o 3 columnas)."""
    return [
        np.array([[volume, zona_id]], dtype=float),
        np.array([[volume, zona_id, 0.0]], dtype=float)
    ]

def _prediction_candidates(volume, zona_id, valores):
    """Candidatos de X para la predicción (para pipelines distintos)."""
    df_full = pd.DataFrame([[volume, zona_id] + list(valores)], columns=CAMPOS_MODELO_DF)
    df_no_zone = pd.DataFrame([[volume] + list(valores)], columns=["Volume"] + CAMPOS_MODELO_DF[2:])
    arr_full = np.array([volume, zona_id] + list(valores), dtype=float).reshape(1, -1)
    arr_no_zone = np.array([volume] + list(valores), dtype=float).reshape(1, -1)
    return [df_full, df_no_zone, arr_full, arr_no_zone]

def predecir_varios_periodos(volume, zona_str, lista_valores, scaler, kmeans, modelos):
    # 1) Zona -> id
    zona_id = ZONAS_MAP[zona_str]

    # 2) Clusterización (robusta a scaler de 2 o 3 features)
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

    # 3) Predicciones por período con fallback de formatos
    preds = []
    for vals in lista_valores:
        y = None
        last = None
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

# =============================
# UI (replica de la app antigua)
# =============================

# --- Descarga de modelos desde URLs ---
with st.expander("⬇️ Descargar modelos desde URLs (GitHub / Drive)", expanded=False):
    st.caption("Pega aquí las URLs de tus artefactos. Se guardarán en la carpeta indicada arriba (por defecto 'modelos').")
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
        base = Path(ruta_modelos)
        try:
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
            st.success("Descarga completada. Ahora puedes cargar los modelos.")
            st.cache_resource.clear()


with st.expander("📦 Descargar un ZIP con TODOS los modelos", expanded=False):
    st.caption("Pega un único enlace a un ZIP (GitHub Release/Drive). La app lo descargará y extraerá los archivos esperados dentro de la carpeta de modelos.")
    zip_url = st.text_input("URL del ZIP (GitHub/Drive)", value="", placeholder="https://github.com/usuario/repo/releases/download/v1.0/modelos.zip")
    if st.button("Descargar y extraer ZIP", use_container_width=True):
        try:
            base = Path(ruta_modelos)
            moved = download_and_extract_zip(zip_url, base)
            if moved:
                st.success("ZIP extraído. Archivos copiados:")
                for m in moved:
                    st.write("•", m)
                st.cache_resource.clear()
            else:
                st.warning("No se encontraron archivos esperados dentro del ZIP. Revisa los nombres requeridos.")
        except Exception as e:
            st.error("No se pudo descargar o extraer el ZIP.")
            st.code(str(e))


        except Exception as e:
            st.error("Falló la descarga de uno o más archivos.")
            st.code(str(e))

try:
    scaler, kmeans, modelos, scaler_name = cargar_modelos(ruta_modelos)
    st.success(f"Modelos cargados desde '{ruta_modelos}' (scaler: {scaler_name}).")
except Exception as e:
    st.error("No se pudieron cargar los modelos.")
    st.code(str(e))
    st.stop()

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

    url = _normalize_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)


# =============================
# Apariencia: fondo
# =============================
def _encode_image(path: Path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background_auto():
    """
    Intenta cargar un archivo llamado 'fondo' con extensiones comunes
    desde ./assets o desde la carpeta del app. Si no existe, aplica
    un gradiente suave por defecto.
    """
    exts = [".jpg", ".jpeg", ".png", ".webp"]
    bases = [Path(__file__).parent / "assets", Path(__file__).parent]
    for base in bases:
        for ext in exts:
            p = base / f"fondo{ext}"
            if p.exists():
                encoded = _encode_image(p)
                st.markdown(
                    f"""
                    <style>
                    .stApp {{
                        background-image: url("data:image/{p.suffix[1:]};base64,{encoded}");
                        background-size: cover;
                        background-attachment: fixed;
                        background-position: center;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                return
    # Fondo por defecto (gradiente)
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, rgba(245,246,250,1) 0%, rgba(255,255,255,1) 60%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_auto()

# =============================
# Configuración
# =============================
st.title("🔍 Predicción de Precio Unitario")
st.caption("Replica de la app original, ajustada para el notebook nuevo.")

# Rutas (como en la app antigua, por defecto el directorio actual)
ruta_modelos = st.text_input("🗂️ Carpeta de modelos (.pkl)", value="modelos")

# Zonas según el notebook (mapeo fijo)
ZONAS = ["Centro", "Norte Chico", "Norte Grande", "Sur"]
ZONAS_MAP = {"Centro": 0, "Norte Chico": 1, "Norte Grande": 2, "Sur": 3}

# Etiquetas legibles para clúster (display)
NOMBRES_CLUSTER = {
    0: "Gran Tronadura",
    1: "Tronadura Fuerte",
    2: "Tronadura Intermedia",
    3: "Tronadura Estándar"
}

# Campos macro (solo para UI). El pipeline del modelo se encarga del escalado.
CAMPOS_MACRO = [
    "CLP/USD",                      # CLP_USD_FX_MONTHLY
    "IPC CL (base 2018)",           # IPC_BASE2018_CP_CL_MONTHLY
    "FERT AMM (Caribe US Spot)",    # FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY
    "FERTECON AGAN BlackSea",       # FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP
    "CPI USA",                      # CPI_USA_MONTHLY
    "ENAP Diesel"                   # FUENTE_ENAP_CHL_DL_MONTHLY
]

# Columnas con nombres del notebook para intentar DataFrame si el modelo lo requiere
CAMPOS_MODELO_DF = [
    "Volume", "Zone",
    "CLP_USD_FX_MONTHLY", "IPC_BASE2018_CP_CL_MONTHLY",
    "FERT_ARGUS_AMM_CARIB_USSPOT_AVG_MTHLY", "FERTECON_AGAN_BLACKSEA_FOB_MTHLY_MP",
    "CPI_USA_MONTHLY", "FUENTE_ENAP_CHL_DL_MONTHLY"
]

# =============================
# Carga de artefactos
# =============================
@st.cache_resource(show_spinner=False)
def cargar_modelos(ruta: str):
    """Carga scaler, kmeans y los 4 modelos por clúster.
    Es compatible con nombres antiguos (scaler.pkl) o nuevos (scaler_cluster.pkl).
    """
    base = Path(ruta)
    # scaler puede llamarse scaler.pkl o scaler_cluster.pkl
    scaler_path = None
    for name in ["scaler.pkl", "scaler_cluster.pkl"]:
        p = base / name
        if p.exists():
            scaler_path = p
            break
    if scaler_path is None:
        raise FileNotFoundError("No se encontró scaler.pkl ni scaler_cluster.pkl en " + str(base))

    kmeans_path = base / "kmeans.pkl"
    if not kmeans_path.exists():
        raise FileNotFoundError("No se encontró kmeans.pkl en " + str(base))

    modelos_paths = {i: base / f"modelo_cluster_{i}_General.pkl" for i in range(4)}
    faltan = [str(p) for p in modelos_paths.values() if not p.exists()]
    if faltan:
        raise FileNotFoundError("Faltan modelos por clúster:\n- " + "\n- ".join(faltan))

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    modelos = {i: joblib.load(modelos_paths[i]) for i in range(4)}
    return scaler, kmeans, modelos, str(scaler_path.name)

def _cluster_transform_candidates(volume, zona_id):
    """Devuelve candidatos de forma para el scaler del clúster (2 o 3 columnas)."""
    return [
        np.array([[volume, zona_id]], dtype=float),
        np.array([[volume, zona_id, 0.0]], dtype=float)
    ]

def _prediction_candidates(volume, zona_id, valores):
    """Candidatos de X para la predicción (para pipelines distintos)."""
    df_full = pd.DataFrame([[volume, zona_id] + list(valores)], columns=CAMPOS_MODELO_DF)
    df_no_zone = pd.DataFrame([[volume] + list(valores)], columns=["Volume"] + CAMPOS_MODELO_DF[2:])
    arr_full = np.array([volume, zona_id] + list(valores), dtype=float).reshape(1, -1)
    arr_no_zone = np.array([volume] + list(valores), dtype=float).reshape(1, -1)
    return [df_full, df_no_zone, arr_full, arr_no_zone]

def predecir_varios_periodos(volume, zona_str, lista_valores, scaler, kmeans, modelos):
    # 1) Zona -> id
    zona_id = ZONAS_MAP[zona_str]

    # 2) Clusterización (robusta a scaler de 2 o 3 features)
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

    # 3) Predicciones por período con fallback de formatos
    preds = []
    for vals in lista_valores:
        y = None
        last = None
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

# =============================
# UI (replica de la app antigua)
# =============================

# --- Descarga de modelos desde URLs ---
with st.expander("⬇️ Descargar modelos desde URLs (GitHub / Drive)", expanded=False):
    st.caption("Pega aquí las URLs de tus artefactos. Se guardarán en la carpeta indicada arriba (por defecto 'modelos').")
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
        base = Path(ruta_modelos)
        try:
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
            st.success("Descarga completada. Ahora puedes cargar los modelos.")
            st.cache_resource.clear()
        except Exception as e:
            st.error("Falló la descarga de uno o más archivos.")
            st.code(str(e))

try:
    scaler, kmeans, modelos, scaler_name = cargar_modelos(ruta_modelos)
    st.success(f"Modelos cargados desde '{ruta_modelos}' (scaler: {scaler_name}).")
except Exception as e:
    st.error("No se pudieron cargar los modelos.")
    st.code(str(e))
    st.stop()

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


# =============================
# Descarga y extracción de ZIP con modelos
# =============================
import io, zipfile

EXPECTED_FILENAMES = {
    "scaler.pkl", "scaler_cluster.pkl",
    "kmeans.pkl",
    "modelo_cluster_0_General.pkl",
    "modelo_cluster_1_General.pkl",
    "modelo_cluster_2_General.pkl",
    "modelo_cluster_3_General.pkl",
}

def download_and_extract_zip(zip_url: str, dest_dir: Path) -> list:
    """Descarga un ZIP desde GitHub/Drive y extrae los archivos en dest_dir.
    Devuelve la lista de archivos relevantes que quedaron en dest_dir.
    """
    url = _normalize_url(zip_url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Descarga en memoria (stream->bytes)
    import requests
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    content = io.BytesIO(r.content)
    moved = []
    with zipfile.ZipFile(content) as z:
        # Extrae TODO a una carpeta temporal dentro de dest_dir
        tmp_extract = dest_dir / "__tmp_zip_extract__"
        if tmp_extract.exists():
            import shutil; shutil.rmtree(tmp_extract)
        z.extractall(tmp_extract)
        # Buscar archivos relevantes en cualquier subcarpeta y moverlos
        for p in tmp_extract.rglob("*"):
            if p.is_file():
                name = p.name
                if name in EXPECTED_FILENAMES:
                    target = dest_dir / name
                    target.write_bytes(p.read_bytes())
                    moved.append(str(target))
        # Limpiar temporal
        import shutil; shutil.rmtree(tmp_extract, ignore_errors=True)
    return moved
