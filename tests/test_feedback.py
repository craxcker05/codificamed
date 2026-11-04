import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

from codificamed.feedback import id_nota, registrar_decision, registrar_latencia

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import monitor  # noqa: E402


def test_feedback_no_guarda_texto(tmp_path):
    nid = id_nota("texto secreto de la nota")
    assert "secreto" not in nid and len(nid) == 10
    registrar_decision(nid, "R51", "gemini", "aceptado", tmp_path)
    registrar_decision(nid, "I10", "gemini", "rechazado", tmp_path)
    filas = list(csv.DictReader((tmp_path / "feedback.csv").open(encoding="utf-8")))
    assert [f["decision"] for f in filas] == ["aceptado", "rechazado"]
    assert set(filas[0]) == {"fecha", "nota_id", "codigo", "proveedor", "decision"}
    with pytest.raises(ValueError):
        registrar_decision(nid, "R51", "gemini", "quizas", tmp_path)


def test_monitor_resumen_y_alertas(tmp_path):
    for i in range(6):
        registrar_decision("n", "R51", "gemini", "aceptado" if i < 2 else "rechazado", tmp_path)
    registrar_latencia("n", "gemini", "m", 200.0, 3, 1, dir_logs=tmp_path)
    por_prov, _ = monitor.resumen_feedback(pd.read_csv(tmp_path / "feedback.csv"))
    assert por_prov.loc["gemini", "aceptacion"] == pytest.approx(0.333, abs=1e-3)
    avisos = monitor.alertas(por_prov, pd.read_csv(tmp_path / "latencia.csv"))
    assert len(avisos) == 2
