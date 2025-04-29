import pathlib
import argparse
import math
from collections import defaultdict
from typing import List
import nltk
from nltk.corpus import stopwords
import pymorphy2

# Если запускаете в первый раз:
# nltk.download('stopwords')
# nltk.download('punkt')

# ------- pymorphy2: Лемматизация запроса -------------------------
morph = pymorphy2.MorphAnalyzer()  # создаём один раз
russian_stopwords = set(stopwords.words('russian'))

# Расширение для списка стоп-слов
extra_stops = {"что", "это", "так", "вот", "быть", "как", "к", "—", "–", "аж", "тыс", "млн", "млрд"}
russian_stopwords |= extra_stops

def lemmatize_query(query: str) -> List[str]:
    # ------- Лемматизация каждого слова в запросе ----------------
    return [morph.parse(word)[0].normal_form for word in query.lower().split()]


# ------- Загрузка tf-idf индекса --------------------------------
def load_index(tfidf_dir: pathlib.Path):
    """
    Загружает tf-idf индекс из указанной директории.
    Возвращает:
        docs: список имён документов (stem'ов)
        index: словарь term/lemma -> список (doc_id, tfidf)
    """
    index = defaultdict(list)
    docs = []

    # ------- Проход по всем файлам с tf-idf ----------------------
    for i, path in enumerate(sorted(tfidf_dir.glob("*.txt"))):

        stem = path.stem.replace("_lemmas_tf_idf", "").replace("_tokens_tf_idf", "")
        docs.append(stem)

        with path.open(encoding="utf-8") as f:
            # ------- Чтение всех строк и парсинг значений ----------
            for line in f:
                if not line.strip():
                    continue
                term, _, tfidf = line.split()
                index[term].append((i, float(tfidf)))

    return docs, index


# ------- Косинусная мера -----------------------------------------
def cosine_similarity(query_vec: dict[str, float],
                      doc_vec: dict[str, float]) -> float:
    """
    Вычисляет косинусную меру между векторами запроса и документа.
    """
    # ------- Скалярное произведение ------------------------------
    dot = sum(query_vec[t] * doc_vec.get(t, 0) for t in query_vec)

    # ------- Нормы векторов запроса и документа ------------------
    norm_query = math.sqrt(sum(v * v for v in query_vec.values()))
    norm_doc   = math.sqrt(sum(v * v for v in doc_vec.values()))

    # ------- Проверка на нулевые векторы -------------------------
    if norm_query == 0 or norm_doc == 0:
        return 0.0
    return dot / (norm_query * norm_doc)


# ------- Векторизация запроса ------------------------------------
def vectorize_query(query_terms: List[str], index: dict[str, List[tuple[int, float]]]):
    """
    Создаёт tf-вектор запроса и строит список затронутых документов.
    """
    tf_counter = defaultdict(int)

    # ------- Подсчёт частотности термов в запросе ----------------
    for term in query_terms:
        tf_counter[term] += 1

    # ------- Нормализация TF -------------------------------------
    total = sum(tf_counter.values())
    tf_query = {t: c / total for t, c in tf_counter.items()}

    # ------- Сбор всех документов, содержащих термы запроса ------
    candidate_docs = set()
    for t in tf_query:
        if t in index:
            candidate_docs.update(doc_id for doc_id, _ in index[t])

    return tf_query, candidate_docs


# ------- Построение векторов документов --------------------------
def build_doc_vectors(index: dict[str, List[tuple[int, float]]],
                      candidate_docs: set[int]) -> dict[int, dict[str, float]]:
    """
    Строит словарь: doc_id -> tf-idf вектор.
    """
    doc_vectors = defaultdict(dict)

    # ------- Проход по индексным данным --------------------------
    for term, postings in index.items():
        # ------- Добавление tf-idf значений только для релевантных документов
        for doc_id, tfidf in postings:
            if doc_id in candidate_docs:
                doc_vectors[doc_id][term] = tfidf

    return doc_vectors


# ------- Основной поиск ------------------------------------------
def search(query: str, docs: List[str], index: dict[str, List[tuple[int, float]]], mode) -> None:
    """
    Выполняет поиск по запросу. Печатает топ-10 совпадений.
    """
    # ------- Преобразование запроса в термы или леммы -----------
    if mode == "lemmas":
        query_terms = lemmatize_query(query)
    else:
        query_terms = query.lower().split()

    # ------- Построение вектора запроса и выбор документов -------
    query_vec, candidate_docs = vectorize_query(query_terms, index)

    # ------- Построение векторов документов ----------------------
    doc_vectors = build_doc_vectors(index, candidate_docs)

    # ------- Вычисление схожести по косинусной мере -------------
    scores = [(doc_id, cosine_similarity(query_vec, doc_vec))
              for doc_id, doc_vec in doc_vectors.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    # ------- Вывод топ-10 документов -----------------------------
    print("Результат поиска:")
    for rank, (doc_id, score) in enumerate(scores[:10], start=1):
        print(f"{rank}. {docs[doc_id]}: {score:.4f}")


# ------- Точка входа и CLI ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Поиск по tf-idf индексу")
    parser.add_argument("--query", required=True,
                        help="Поисковый запрос (разделённый пробелами)")
    parser.add_argument("--mode", choices=["tokens", "lemmas"], default="lemmas",
                        help="Режим поиска: tokens или lemmas (по умолчанию lemmas)")
    args = parser.parse_args()

    # ------- Определение директории индекса ----------------------
    tfidf_dir = pathlib.Path("lemmas_tf_idf" if args.mode == "lemmas" else "tokens_tf_idf")

    # ------- Загрузка индекса и запуск поиска --------------------
    docs, index = load_index(tfidf_dir)
    search(args.query, docs, index, args.mode)


if __name__ == "__main__":
    main()
