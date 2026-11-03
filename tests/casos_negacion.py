"""Frases de prueba para negación: (frase, mención, estado esperado).

Son frases inventadas para este proyecto (no hay historias reales). Se
escribieron junto con las reglas, así que miden regresión y cobertura de
patrones, no el desempeño sobre notas reales.
"""

A, N, S = "afirmado", "negado", "sospecha"

CASOS: list[tuple[str, str, str]] = [
    # Los cinco ejemplos exigidos
    ("El paciente niega fiebre.", "fiebre", N),
    ("Se descarta neumonía.", "neumonía", N),
    ("Sin signos de TVP.", "TVP", N),
    ("No HTA.", "HTA", N),
    ("DM2 en tratamiento.", "DM2", A),
    # Negación previa
    ("Niega dolor torácico ni disnea.", "disnea", N),
    ("No refiere náuseas ni vómitos.", "vómitos", N),
    ("Paciente sin antecedentes de diabetes.", "diabetes", N),
    ("Ausencia de edema en miembros inferiores.", "edema", N),
    ("No se evidencia derrame pleural en la radiografía.", "derrame pleural", N),
    ("Cultivo negativo para tuberculosis.", "tuberculosis", N),
    ("Abdomen sin masas palpables.", "masas", N),
    ("No presenta cefalea.", "cefalea", N),
    ("Niega tos, fiebre y disnea.", "disnea", N),
    ("Sin fiebre ni escalofríos desde ayer.", "escalofríos", N),
    # Negación posterior
    ("Neumonía descartada.", "Neumonía", N),
    ("Infarto agudo de miocardio descartado por troponinas.", "Infarto agudo de miocardio", N),
    ("Embolia pulmonar negada por el paciente.", "Embolia pulmonar", N),
    # Sospecha
    ("Probable faringitis viral.", "faringitis viral", S),
    ("Sospecha de apendicitis aguda.", "apendicitis aguda", S),
    ("Posible infección urinaria.", "infección urinaria", S),
    ("Cuadro compatible con gastroenteritis.", "gastroenteritis", S),
    ("Se solicita tomografía para descartar embolia pulmonar.", "embolia pulmonar", S),
    ("No se descarta neumonía.", "neumonía", S),
    ("Bronquitis probable.", "Bronquitis", S),
    # Afirmado
    ("Presenta fiebre de 39 grados desde hace 3 días.", "fiebre", A),
    ("Paciente con hipertensión arterial en tratamiento.", "hipertensión arterial", A),
    ("Diagnóstico: faringitis aguda.", "faringitis aguda", A),
    ("Refiere dolor abdominal y vómitos.", "vómitos", A),
    ("Antecedente de asma bronquial desde la infancia.", "asma bronquial", A),
    # Terminadores y alcance
    ("Niega fiebre pero presenta tos.", "tos", A),
    ("Sin dolor torácico, refiere disnea de esfuerzo.", "disnea", A),
    ("No hay fiebre; presenta cefalea intensa.", "cefalea", A),
    ("Niega fiebre. Presenta tos productiva.", "tos", A),
    ("Probable neumonía, se descarta embolia pulmonar.", "embolia pulmonar", N),
    ("Probable neumonía, se descarta embolia pulmonar.", "neumonía", S),
    # Pseudo-negaciones
    ("Sin embargo, presenta fiebre alta.", "fiebre", A),
    ("No solo tiene tos, también presenta disnea.", "disnea", A),
    ("Sin duda se trata de una neumonía.", "neumonía", A),
    # Casos con abreviaturas ya expandidas
    ("Sin signos de trombosis venosa profunda.", "trombosis venosa profunda", N),
    ("No hipertensión arterial.", "hipertensión arterial", N),
    ("Niega antecedentes de diabetes mellitus tipo 2.", "diabetes mellitus tipo 2", N),
    # Provenientes de fallos hallados en las notas ficticias
    ("Dx: faringitis aguda, probable etiología viral.", "faringitis aguda", A),
    ("Tampoco dolor torácico.", "dolor torácico", N),
]


def evaluar_ficticias() -> list[tuple[str, str, str, str]]:
    """Aplica pipeline completo a data/ficticios/notas_ficticias.jsonl.

    Cada frase_justificacion debe salir 'afirmado' y la última palabra de cada
    frase de `negados` debe salir 'negado'. Devuelve (id, frase, esperado,
    obtenido) de todos los casos evaluados.
    """
    import json
    from pathlib import Path

    from codificamed.negation import clasificar_en_texto
    from codificamed.preprocess import preprocesar

    ruta = Path(__file__).resolve().parents[1] / "data" / "ficticios" / "notas_ficticias.jsonl"
    filas = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        nota = json.loads(linea)
        tp = preprocesar(nota["texto"])

        def rango(frase: str, solo_ultima: bool) -> tuple[int, int]:
            a = nota["texto"].index(frase)
            idx = [i for i in range(len(tp.texto))
                   if tp._orig_ini[i] >= a and tp._orig_fin[i] <= a + len(frase)]
            ini, fin = idx[0], idx[-1] + 1
            if solo_ultima:
                ini = ini + len(tp.texto[ini:fin].rstrip()) - len(tp.texto[ini:fin].split()[-1])
            return ini, fin

        for frase in nota["frase_justificacion"].values():
            ini, fin = rango(frase, False)
            filas.append((nota["id"], frase, "afirmado", clasificar_en_texto(tp, ini, fin).estado.value))
        for frase in nota["negados"].values():
            ini, fin = rango(frase, True)
            filas.append((nota["id"], frase, "negado", clasificar_en_texto(tp, ini, fin).estado.value))
    return filas
