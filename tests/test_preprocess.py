import pytest

from codificamed.preprocess import normalizar, preprocesar


def test_normalizar_conserva_longitud() -> None:
    t = "a b\tc\r\nd “hola”"
    assert len(normalizar(t)) == len(t)


def test_expande_abreviaturas_y_conserva_offsets() -> None:
    original = "Pcte con APP de HTA y DM2 en tto."
    tp = preprocesar(original)
    assert tp.texto.startswith("Paciente con") and "hipertensión arterial" in tp.texto
    ini = tp.texto.index("hipertensión arterial")
    assert tp.evidencia(ini, ini + len("hipertensión arterial")) == "HTA"
    ini = tp.texto.index("tratamiento")
    assert tp.evidencia(ini, ini + len("tratamiento")) == "tto"


def test_evidencia_parcial_de_expansion_cubre_toda_la_abreviatura() -> None:
    tp = preprocesar("Con HTA.")
    ini = tp.texto.index("arterial")
    assert tp.evidencia(ini, ini + 3) == "HTA"


def test_oraciones_mapean_a_texto_original() -> None:
    original = "Sr. Pérez, 58 a., APP: HTA, DM2.\nTA 150/95, FC 78 lpm. Rx de tórax normal."
    tp = preprocesar(original)
    oraciones = tp.oraciones()
    assert len(oraciones) >= 2
    for o in oraciones:
        assert 0 <= o.ini_orig < o.fin_orig <= len(original)


def test_contextuales_solo_con_numero() -> None:
    tp = preprocesar("TA 150/95 y 3 d de evolución. La TA es baja. Hace 2 h. IC y CC.")
    assert "Tensión arterial 150/95" in tp.texto
    assert "3 días" in tp.texto and "2 horas" in tp.texto
    assert "La TA es baja" in tp.texto  # sin número: no se expande
    assert "IC y CC" in tp.texto  # ambiguas: nunca


def test_minusculas_y_palabras_comunes() -> None:
    tp = preprocesar("hta conocida. Pasó por el eco del valle.")
    assert "hipertensión arterial conocida" in tp.texto
    assert "eco del valle" in tp.texto  # "eco" en minúscula no es ECO


def test_no_expande_dentro_de_palabras() -> None:
    assert preprocesar("Hbpatología HTAS ivan").texto == "Hbpatología HTAS ivan"


def test_segmentacion_basica() -> None:
    tp = preprocesar("Niega fiebre. Presenta tos? Sí! Tiene 3.5 mg de algo.")
    assert [o.texto for o in tp.oraciones()] == [
        "Niega fiebre.", "Presenta tos?", "Sí!", "Tiene 3.5 mg de algo."]


def test_segmentacion_no_corta_en_abreviaturas() -> None:
    tp = preprocesar("Dr. López indica aprox. cinco días. Control en una semana.")
    assert [o.texto for o in tp.oraciones()] == [
        "Dr. López indica aprox. cinco días.", "Control en una semana."]


def test_salto_simple_de_maquetacion_no_corta() -> None:
    tp = preprocesar("Paciente con tos\nproductiva y fiebre. Alta.")
    assert tp.oraciones()[0].texto == "Paciente con tos\nproductiva y fiebre."


def test_rango_invalido() -> None:
    with pytest.raises(ValueError):
        preprocesar("hola").span_original(3, 2)
