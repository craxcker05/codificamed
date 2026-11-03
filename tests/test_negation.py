from codificamed.negation import Estado, clasificar_en_texto, clasificar_frase
from codificamed.preprocess import preprocesar
from tests.casos_negacion import CASOS, evaluar_ficticias


def test_hay_al_menos_25_frases_con_los_ejemplos_exigidos() -> None:
    assert len(CASOS) >= 25
    frases = {f for f, _, _ in CASOS}
    for exigida in ("El paciente niega fiebre.", "Se descarta neumonía.",
                    "Sin signos de TVP.", "No HTA.", "DM2 en tratamiento."):
        assert exigida in frases


def test_cada_frase_obtiene_su_estado() -> None:
    fallos = []
    for frase, mencion, esperado in CASOS:
        obtenido = clasificar_frase(frase, mencion).estado.value
        if obtenido != esperado:
            fallos.append((frase, mencion, esperado, obtenido))
    assert not fallos, fallos


def test_ejemplos_exigidos_sin_puntuacion() -> None:
    assert clasificar_frase("niega fiebre", "fiebre").estado is Estado.NEGADO
    assert clasificar_frase("DM2 en tratamiento", "DM2").estado is Estado.AFIRMADO


def test_devuelve_el_disparador() -> None:
    r = clasificar_frase("Se descarta neumonía.", "neumonía")
    assert r.disparador == "Se descarta" and r.ini_disparador == 0


def test_integracion_con_texto_procesado() -> None:
    tp = preprocesar("Pcte con APP de HTA. Niega dolor torácico. Sin signos de TVP.")
    for mencion, esperado in (("hipertensión arterial", Estado.AFIRMADO),
                              ("dolor torácico", Estado.NEGADO),
                              ("trombosis venosa profunda", Estado.NEGADO)):
        ini = tp.texto.index(mencion)
        assert clasificar_en_texto(tp, ini, ini + len(mencion)).estado is esperado


def test_notas_ficticias_extremo_a_extremo() -> None:
    fallos = [f for f in evaluar_ficticias() if f[2] != f[3]]
    assert not fallos, fallos
