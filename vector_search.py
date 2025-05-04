import pathlib
import argparse
import math
from collections import defaultdict
from typing import List, Dict, Set
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import pymorphy2
from SearchResult import SearchResult

# Если запускаете в первый раз:
# nltk.download('stopwords')
# nltk.download('punkt')

# ------- pymorphy2: Лемматизация запроса -------------------------
morph = pymorphy2.MorphAnalyzer()  # создаём один раз
russian_stopwords = set(stopwords.words('russian'))

# Расширение для списка стоп-слов
extra_stops = {"что", "это", "так", "вот", "быть", "как", "к", "—", "–", "аж", "тыс", "млн", "млрд"}
russian_stopwords |= extra_stops

ALPHA = 0.8  # значимость векторного score
HTML_DIR = pathlib.Path("links")  # где лежат *.html
MAPPINGS_FILE = pathlib.Path("index.txt")  # где лежит файл с привязками "файл - url"


def lemmatize_query(query: str) -> List[str]:
    # ------- Лемматизация каждого слова в запросе ----------------
    return [morph.parse(word)[0].normal_form for word in query.lower().split()]


# Построение зависимостей между страницами
def build_link_graph(docs: List[str]) -> Dict[str, Set[str]]:
    """
    Возвращает граф ссылок: src_id -> {dst_id, …}.
    Учитываются только ссылки, у которых <a href="*.html"> ведёт
    на один из локальных документов.
    """
    graph: Dict[str, Set[str]] = defaultdict(set)

    for src_id, name in enumerate(docs):
        html_path = HTML_DIR / f"{name}.html"
        if not html_path.exists():
            continue

        with html_path.open(encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href in reverse_mappings:
                graph[name].add(reverse_mappings[href])

    # гарантируем наличие ключа для каждой вершины
    for doc in docs:
        graph.setdefault(doc, set())
    return graph


# Вычисление page rank
def compute_pagerank(graph: Dict[str, Set[str]],
                     damping: float = 0.85,
                     iters: int = 20) -> Dict[str, float]:
    pages = list(graph.keys())
    index = {page: i for i, page in enumerate(pages)}
    n = len(pages)

    pr = [1.0 / n] * n
    out_deg = {page: len(links) for page, links in graph.items()}

    for _ in range(iters):
        new_pr = [0.0] * n
        for page, links in graph.items():
            u = index[page]
            share = pr[u] / out_deg[page] if out_deg[page] else 0.0
            for linked_page in links:
                if linked_page in index:
                    v = index[linked_page]
                    new_pr[v] += damping * share
        leak = 1.0 - sum(new_pr)
        new_pr = [score + leak / n for score in new_pr]
        pr = new_pr

    return {page: pr[i] for page, i in index.items()}


# -----------------------------------------------------------------

def load_url_mappings(reverse):
    result = {}
    with open(MAPPINGS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '-' not in line:
                continue  # Skip empty or malformed lines
            key, value = map(str.strip, line.split('-', 1))
            result[value if reverse else key] = key if reverse else value
    return result


mappings = load_url_mappings(reverse=False)
reverse_mappings = load_url_mappings(reverse=True)


def get_page_data_by_saved_html_name(saved_html_name: str) -> (str, str):
    html_path = HTML_DIR / f"{saved_html_name}.html"
    with html_path.open(encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    title = soup.find("meta", attrs={"property": "og:title"})

    return title["content"].strip(), mappings[saved_html_name]


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
    norm_doc = math.sqrt(sum(v * v for v in doc_vec.values()))

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
def search(query: str, mode: str) -> List[SearchResult]:
    """
    Выполняет поиск по запросу. Печатает топ-10 совпадений.
    """
    # ------- Определение директории индекса ----------------------
    tfidf_dir = pathlib.Path("lemmas_tf_idf" if mode == "lemmas" else "tokens_tf_idf")

    # ------- Загрузка индекса и запуск поиска --------------------
    docs, index = load_index(tfidf_dir)

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
    sim_scores = {doc_id: cosine_similarity(query_vec, doc_vec)
                  for doc_id, doc_vec in doc_vectors.items()}

    # ---------- 2. PageRank по пулу cand (можно и по всем) --
    link_graph = build_link_graph(docs)
    pagerank_all = compute_pagerank(link_graph)
    pr_scores = {doc_id: pagerank_all[docs[doc_id]] for doc_id in candidate_docs}

    # ---------- 3. min-max нормализация ---------------------------
    def min_max(values: Dict[int, float]) -> Dict[int, float]:
        if not values:
            return {}
        vmin, vmax = min(values.values()), max(values.values())
        if vmax - vmin < 1e-12:
            return {k: 0.0 for k in values}
        return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}

    sim_norm = min_max(sim_scores)
    pr_norm = min_max(pr_scores)

    # ---------- 4. финальный счёт ---------------------------------
    final = [(doc_id, ALPHA * sim_norm.get(doc_id, 0.0) +
              (1 - ALPHA) * pr_norm.get(doc_id, 0.0))
             for doc_id in candidate_docs]
    final.sort(key=lambda x: x[1], reverse=True)

    top_ten_results = []

    for doc_id, score in final:
        (web_page_name, web_page_url) = get_page_data_by_saved_html_name(docs[doc_id])

        top_ten_results.append(
            SearchResult(
                web_page_name=web_page_name,
                web_page_url=web_page_url,
                saved_html_name=docs[doc_id],
                similarity=sim_scores.get(doc_id, 0.0),
                page_rank=pr_scores.get(doc_id, 0.0),
                score=score,
            )
        )

    return top_ten_results


# ------- Точка входа и CLI ---------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Поиск по tf-idf индексу")
    parser.add_argument("--query", required=True,
                        help="Поисковый запрос (разделённый пробелами)")
    parser.add_argument("--mode", choices=["tokens", "lemmas"], default="lemmas",
                        help="Режим поиска: tokens или lemmas (по умолчанию lemmas)")
    args = parser.parse_args()

    search_results = search(args.query, args.mode)

    print("\nРезультат поиска:")
    for rank, search_result in enumerate(search_results[:10], start=1):
        print(f"{rank:02}. {search_result.saved_html_name}:"
              f"    Similarity={search_result.similarity:.4f}"
              f"    PageRank={search_result.page_rank:.10f}"
              f"    Score={search_result.score:.3f}")


if __name__ == "__main__":
    main()
