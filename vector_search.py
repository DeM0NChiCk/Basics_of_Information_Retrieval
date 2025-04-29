import argparse
import pathlib
import math
from collections import defaultdict
from typing import Dict, List, Tuple

def load_tfidf_vectors(tfidf_dir: pathlib.Path) -> Tuple[Dict[str, Dict[str, float]], set]:
    """
    Загружает TF-IDF-вектора документов.
    Возвращает:
      - словарь: {stem: {term: tfidf}}
      - множество всех термов
    """
    vectors = {}
    vocabulary = set()

    for path in tfidf_dir.glob("*.txt"):
        stem = path.stem.replace("_tokens_tf_idf", "").replace("_lemmas_tf_idf", "")
        with path.open(encoding="utf-8") as f:
            tfidf_vector = {}
            for line in f:
                term, _, tfidf = line.strip().split()
                tfidf = float(tfidf)
                tfidf_vector[term] = tfidf
                vocabulary.add(term)
            vectors[stem] = tfidf_vector

    return vectors, vocabulary


def build_query_vector(query_terms: List[str], doc_vectors: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Строит вектор запроса: tf = 1, idf берётся из любого документа, где есть термин.
    """
    query_vector = {}
    doc_count = len(doc_vectors)
    df = defaultdict(int)

    for term in query_terms:
        for vec in doc_vectors.values():
            if term in vec:
                df[term] += 1

    for term in query_terms:
        if df[term] > 0:
            idf = math.log(doc_count / df[term])
            query_vector[term] = idf  # tf=1

    return query_vector


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    common_terms = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[t] * vec2[t] for t in common_terms)

    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return numerator / (norm1 * norm2)


def search(query: str, tfidf_dir: pathlib.Path, use_lemmas: bool = False, top_k: int = 5):
    vectors, _ = load_tfidf_vectors(tfidf_dir)
    query_terms = query.lower().split()
    query_vec = build_query_vector(query_terms, vectors)

    scores = []
    for stem, vec in vectors.items():
        score = cosine_similarity(query_vec, vec)
        if score > 0:
            scores.append((stem, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔍 Top {top_k} результатов по запросу: '{query}'")
    for rank, (stem, score) in enumerate(scores[:top_k], start=1):
        print(f"{rank}. {stem} — score: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Поиск по TF-IDF индексам")
    parser.add_argument("query", help="Поисковый запрос")
    parser.add_argument("--tfidf-dir", default="tokens_tf_idf", type=pathlib.Path,
                        help="Каталог с TF-IDF файлами")
    parser.add_argument("--use-lemmas", action="store_true", help="Использовать леммы вместо терминов")
    parser.add_argument("--top-k", type=int, default=10, help="Сколько результатов выводить")
    args = parser.parse_args()

    tfidf_dir = pathlib.Path("lemmas_tf_idf" if args.use_lemmas else "tokens_tf_idf")
    search(args.query, tfidf_dir=tfidf_dir, top_k=args.top_k)


if __name__ == "__main__":
    main()
