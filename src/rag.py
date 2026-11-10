"""Motor de recuperación documental para el agente de calidad del aire.

Busca fragmentos relevantes en un corpus local utilizando TF-IDF
y similitud coseno. El LLM del agente genera la respuesta final.
"""

import json
import re
import unicodedata
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DOCS_DIR

CORPUS = DOCS_DIR / "fuentes.json"


def _normalizar(texto: str) -> str:
    """Normaliza mayúsculas, tildes y espacios."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", texto).strip()


@lru_cache(maxsize=1)
def _indice():
    """Carga los documentos y construye el índice TF-IDF."""

    with CORPUS.open(encoding="utf-8") as archivo:
        documentos = json.load(archivo)

    if not documentos:
        raise ValueError("El corpus RAG no contiene documentos")

    ids = [doc["id"] for doc in documentos]

    if len(ids) != len(set(ids)):
        raise ValueError("Hay identificadores duplicados")

    textos = [
        _normalizar(
            f'{doc["titulo"]} {doc["seccion"]} {doc["texto"]}'
        )
        for doc in documentos
    ]

    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        max_df=1.0,
    )

    matriz = vectorizador.fit_transform(textos)

    return documentos, vectorizador, matriz


def search(query: str, k: int = 4) -> list[dict]:
    """Recupera documentos relevantes para la pregunta."""

    if not isinstance(query, str) or not query.strip():
        return []

    k = max(1, min(int(k), 6))

    documentos, vectorizador, matriz = _indice()

    vector_pregunta = vectorizador.transform([
        _normalizar(query)
    ])

    similitudes = cosine_similarity(
        vector_pregunta, matriz
    ).ravel()

    indices = similitudes.argsort()[::-1]

    resultados = []

    for i in indices:
        if similitudes[i] < 0.045:
            break

        documento = documentos[int(i)]

        resultados.append({
            "id": documento["id"],
            "titulo": documento["titulo"],
            "seccion": documento["seccion"],
            "fuente_url": documento["url"],
            "fragmento": documento["texto"],
            "similitud": round(float(similitudes[i]), 3),
        })

        if len(resultados) >= k:
            break

    return resultados


if __name__ == "__main__":
    import sys

    pregunta = " ".join(sys.argv[1:]) or (
        "¿Cuál es el límite diario de PM2.5 en Colombia?"
    )

    print(json.dumps(
        search(pregunta),
        ensure_ascii=False,
        indent=2,
    ))
