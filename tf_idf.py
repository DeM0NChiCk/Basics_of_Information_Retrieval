import argparse
import math
import pathlib
from collections import Counter, defaultdict


def collect_documents(html_dir: pathlib.Path,
                      tokens_dir: pathlib.Path,
                      lemmas_dir: pathlib.Path):
    """
    Возвращает список кортежей
        (stem, html_path, tokens_path, lemmas_path)
    только для тех базовых имён (stem), которые присутствуют во
    **всех трёх** директориях.
    """
    html_map   = {p.stem: p for p in html_dir.glob("*.html")}
    tokens_map = {p.stem[len("tokens_"):]: p for p in tokens_dir.glob("*.txt")}
    lemmas_map = {p.stem[len("lemmas_"):]: p for p in lemmas_dir.glob("*.txt")}

    common_stems = html_map.keys() & tokens_map.keys() & lemmas_map.keys()
    if not common_stems:
        raise FileNotFoundError(
            "Не найдено ни одного набора html/tokens/lemmas с общим именем"
        )

    for stem in sorted(common_stems):
        yield stem, html_map[stem], tokens_map[stem], lemmas_map[stem]


def compute_tfidf(html_dir: pathlib.Path,
                  tokens_dir: pathlib.Path,
                  lemmas_dir: pathlib.Path) -> None:
    terms_out_dir = pathlib.Path("tokens_tf_idf")
    lemmas_out_dir = pathlib.Path("lemmas_tf_idf")
    terms_out_dir.mkdir(exist_ok=True)
    lemmas_out_dir.mkdir(exist_ok=True)

    # ---------- 1. Собираем списки документов ----------------------------------------
    docs = list(collect_documents(html_dir, tokens_dir, lemmas_dir))
    N = len(docs)

    # ---------- 2. Первый проход: счётчики df ---------------------------------------
    term_docs:  dict[str, set[int]] = defaultdict(set)
    lemma_docs: dict[str, set[int]] = defaultdict(set)

    for doc_id, (stem, _, tokens_path, lemmas_path) in enumerate(docs):
        # Термины
        with tokens_path.open(encoding="utf‑8") as f:
            for term in {line.strip() for line in f if line.strip()}:
                term_docs[term].add(doc_id)

        # Леммы
        with lemmas_path.open(encoding="utf‑8") as f:
            for lemma in {line.split(maxsplit=1)[0] for line in f if line.strip()}:
                lemma_docs[lemma].add(doc_id)

    idf_term  = {t: math.log(N / len(docs)) for t, docs in term_docs.items()}
    idf_lemma = {l: math.log(N / len(docs)) for l, docs in lemma_docs.items()}

    # ---------- 3. Второй проход: tf и запись ---------------------------------------
    for doc_id, (stem, html_path, tokens_path, lemmas_path) in enumerate(docs):
        # --- tf терминов ------------------------------------------------------------
        with tokens_path.open(encoding="utf‑8") as f:
            token_list = [line.strip() for line in f if line.strip()]
        term_counter = Counter(token_list)
        total_terms  = len(token_list)

        # --- tf лемм ---------------------------------------------------------------
        lemma_counter: Counter[str] = Counter()
        with lemmas_path.open(encoding="utf‑8") as f:
            for line in f:
                if not line.strip():
                    continue
                lemma, *lemma_terms = line.split()
                freq = sum(term_counter[t] for t in lemma_terms)
                if freq:
                    lemma_counter[lemma] = freq

        # --- вывод ----------------------------------------------------------------
        out_terms = terms_out_dir / f"{stem}_tokens_tf_idf.txt"
        out_lemmas = lemmas_out_dir / f"{stem}_lemmas_tf_idf.txt"

        with out_terms.open("w", encoding="utf‑8") as out:
            for term, cnt in term_counter.items():
                tf = cnt / total_terms
                idf = idf_term[term]
                out.write(f"{term} {idf:.6f} {tf*idf:.6f}\n")

        with out_lemmas.open("w", encoding="utf‑8") as out:
            for lemma, cnt in lemma_counter.items():
                tf = cnt / total_terms
                idf = idf_lemma[lemma]
                out.write(f"{lemma} {idf:.6f} {tf*idf:.6f}\n")


# ---------- CLI ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Считает tf‑idf для html‑коллекции с разнесёнными каталогами"
    )
    parser.add_argument("--html-dir",   default="links",   type=pathlib.Path,
                        help="Каталог, где лежат *.html (по умолчанию links/)")
    parser.add_argument("--tokens-dir", default="tokens",  type=pathlib.Path,
                        help="Каталог с *.txt, содержащими список терминов")
    parser.add_argument("--lemmas-dir", default="lemmas",  type=pathlib.Path,
                        help="Каталог с *.txt, содержащими леммы и их термины")
    args = parser.parse_args()

    compute_tfidf(args.html_dir, args.tokens_dir, args.lemmas_dir)


if __name__ == "__main__":
    main()
