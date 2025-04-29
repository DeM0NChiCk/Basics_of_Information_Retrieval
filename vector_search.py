import pathlib
import argparse
import math
from collections import defaultdict
from typing import List

import pymorphy2
morph = pymorphy2.MorphAnalyzer()  # создаём один раз

def lemmatize_query(query: str) -> List[str]:
    return [morph.parse(word)[0].normal_form for word in query.lower().split()]


def load_index(tfidf_dir: pathlib.Path):
    """
    Загружает tf-idf индекс из указанной директории.
    Возвращает:
        docs: список имён документов (stem'ов)
        index: словарь term/lemma -> список (doc_id, tfidf)
    """
    index = defaultdict(list)
    docs = []

    for i, path in enumerate(sorted(tfidf_dir.glob("*.txt"))):
        docs.append(path.stem)
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                term, _, tfidf = line.split()
                index[term].append((i, float(tfidf)))

    return docs, index


def cosine_similarity(query_vec: dict[str, float],
                      doc_vec: dict[str, float]) -> float:
    """
    Вычисляет косинусную меру между векторами запроса и документа.
    """
    dot = sum(query_vec[t] * doc_vec.get(t, 0) for t in query_vec)
    norm_query = math.sqrt(sum(v * v for v in query_vec.values()))
    norm_doc   = math.sqrt(sum(v * v for v in doc_vec.values()))
    if norm_query == 0 or norm_doc == 0:
        return 0.0
    return dot / (norm_query * norm_doc)


def vectorize_query(query_terms: List[str], index: dict[str, List[tuple[int, float]]]):
    """
    Создаёт tf-вектор запроса и строит список затронутых документов.
    """
    tf_counter = defaultdict(int)
    for term in query_terms:
        tf_counter[term] += 1

    total = sum(tf_counter.values())
    tf_query = {t: c / total for t, c in tf_counter.items()}

    # Вычисляем вектор документа для всех doc_id, где встречались термины
    candidate_docs = set()
    for t in tf_query:
        if t in index:
            candidate_docs.update(doc_id for doc_id, _ in index[t])

    return tf_query, candidate_docs


def build_doc_vectors(index: dict[str, List[tuple[int, float]]],
                      candidate_docs: set[int]) -> dict[int, dict[str, float]]:
    """
    Строит словарь: doc_id -> tf-idf вектор.
    """
    doc_vectors = defaultdict(dict)
    for term, postings in index.items():
        for doc_id, tfidf in postings:
            if doc_id in candidate_docs:
                doc_vectors[doc_id][term] = tfidf
    return doc_vectors


def search(query: str, docs: List[str], index: dict[str, List[tuple[int, float]]], mode) -> None:
    """
    Выполняет поиск по запросу. Печатает топ-10 совпадений.
    """
    if mode == "lemmas":
        query_terms = lemmatize_query(query)
    else:
        query_terms = query.lower().split()
    query_vec, candidate_docs = vectorize_query(query_terms, index)
    doc_vectors = build_doc_vectors(index, candidate_docs)

    scores = [(doc_id, cosine_similarity(query_vec, doc_vec))
              for doc_id, doc_vec in doc_vectors.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    print("Топ совпадений:")
    for doc_id, score in scores[:10]:
        print(f"{docs[doc_id]}: {score:.4f}")


# ---------- CLI ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Поиск по tf-idf индексу")
    parser.add_argument("--query", required=True,
                        help="Поисковый запрос (разделённый пробелами)")
    parser.add_argument("--mode", choices=["tokens", "lemmas"], default="lemmas",
                        help="Режим поиска: tokens или lemmas (по умолчанию lemmas)")
    args = parser.parse_args()

    tfidf_dir = pathlib.Path("lemmas_tf_idf" if args.mode == "lemmas" else "tokens_tf_idf")

    docs, index = load_index(tfidf_dir)
    search(args.query, docs, index, args.mode)


if __name__ == "__main__":
    main()
