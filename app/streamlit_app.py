"""CodificaMed: sugerencia de códigos CIE-10 con evidencia, feedback y registro de latencia.

Uso: streamlit run app/streamlit_app.py
Prototipo académico: no usar con historias clínicas reales.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from codificamed.feedback import id_nota, registrar_decision, registrar_latencia  # noqa: E402
from codificamed.llm import crear_proveedor  # noqa: E402
from codificamed.pipeline import analizar  # noqa: E402
from codificamed.schemas import ResultadoNota  # noqa: E402
from config.settings import get_settings  # noqa: E402

st.set_page_config(page_title="CodificaMed", layout="wide")
st.title("CodificaMed — sugerencia de códigos CIE-10")
st.warning("Prototipo académico. Usar solo notas ficticias o de CodiEsp; nunca historias clínicas reales.")

ficticias = [json.loads(l) for l in (RAIZ / "data/ficticios/notas_ficticias.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]

with st.sidebar:
    proveedores = ["ollama", "gemini"]
    por_defecto = proveedores.index(get_settings().llm_provider)
    proveedor = st.selectbox("Proveedor de LLM", proveedores, index=por_defecto)
    ejemplo = st.selectbox("Nota de ejemplo", [n["id"] for n in ficticias])
    if st.button("Cargar ejemplo"):
        st.session_state["nota"] = next(n["texto"] for n in ficticias if n["id"] == ejemplo)
        st.session_state["id_ejemplo"] = ejemplo
        st.session_state.pop("resultado", None)

nota = st.text_area("Nota clínica", key="nota", height=200)


def resaltar(texto: str, res: ResultadoNota) -> str:
    """Verde = confirmado, gris tachado = negado/sospecha (las evidencias no se solapan)."""
    marcas = sorted([(s.offsets[0], s.offsets[1], s.estado) for s in [*res.sugerencias, *res.negados]])
    out, pos = "", 0
    for ini, fin, estado in marcas:
        if ini < pos:
            continue
        estilo = ("background:#c8f7c5" if estado == "afirmado"
                  else "background:#e0e0e0;text-decoration:line-through")
        out += html.escape(texto[pos:ini]) + f'<mark style="{estilo}">{html.escape(texto[ini:fin])}</mark>'
        pos = fin
    return (out + html.escape(texto[pos:])).replace("\n", "<br>")


if st.button("Codificar", type="primary", disabled=not nota.strip()):
    nid = id_nota(nota, st.session_state.get("id_ejemplo"))
    try:
        with st.spinner(f"Procesando con {proveedor}…"):
            ajustes = get_settings().model_copy(update={"llm_provider": proveedor})
            res = analizar(nota, crear_proveedor(ajustes))
        registrar_latencia(nid, res.proveedor, res.modelo, res.tiempo_total_s,
                           res.llamadas_llm, len(res.sugerencias))
        st.session_state["resultado"] = (res, nota, nid)
    except Exception as e:  # noqa: BLE001 - se registra y se muestra, sin claves
        registrar_latencia(nid, proveedor, "", 0.0, 0, 0, error=type(e).__name__ + ": " + str(e))
        st.error(f"Error al procesar: {type(e).__name__}. Ver logs/latencia.csv")

if "resultado" in st.session_state:
    res, texto, nid = st.session_state["resultado"]
    st.caption(f"{res.proveedor} / {res.modelo} — tiempo real medido: {res.tiempo_total_s} s, "
               f"{res.llamadas_llm} llamadas al LLM")
    st.markdown(resaltar(texto, res), unsafe_allow_html=True)
    st.subheader("Códigos sugeridos")
    if not res.sugerencias:
        st.info("Sin códigos sugeridos.")
    for s in res.sugerencias:
        c1, c2, c3, c4 = st.columns([1, 4, 3, 2])
        c1.markdown(f"**{s.codigo_original}**")
        c2.write(s.descripcion)
        c3.write(f"«{s.evidencia}» · confianza {s.confianza:.2f}")
        if c4.button("Aceptar", key=f"a_{nid}_{s.codigo}"):
            registrar_decision(nid, s.codigo_original, res.proveedor, "aceptado")
            c4.success("Registrado")
        if c4.button("Rechazar", key=f"r_{nid}_{s.codigo}"):
            registrar_decision(nid, s.codigo_original, res.proveedor, "rechazado")
            c4.warning("Registrado")
    if res.negados:
        st.subheader("Negados o con sospecha (no se sugieren)")
        st.table([{"evidencia": n.evidencia, "estado": n.estado} for n in res.negados])
