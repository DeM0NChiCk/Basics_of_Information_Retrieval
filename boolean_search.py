import os
import glob
import nltk
from nltk.corpus import stopwords
import pymorphy2

# Если запускаете в первый раз:
# nltk.download('stopwords')
# nltk.download('punkt')

# Инициализируем морфологический анализатор и стоп-слова
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

# Расширение для списка стоп-слов
extra_stops = {"что", "это", "так", "вот", "быть", "как", "к", "—", "–", "аж", "тыс", "млн", "млрд"}
russian_stopwords |= extra_stops

def build_inverted_index(lemmas_dir: str, input_file: str, output_file: str) -> dict[str, set]:
    """
    Строит инвертированный индекс и сохраняет его в файл.
    """
    inverted_index = {}

    # Проходим по всем файлам с леммами в указанной директории
    for lemma_file in glob.glob(os.path.join(lemmas_dir, input_file)):
        # Получаем имя документа
        doc_id = os.path.splitext(os.path.basename(lemma_file))[0].replace("lemmas_", "")

        # Открываем файл и читаем его построчно
        with open(lemma_file, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue # Пропускаем строки без леммы
                lemma = line.split(":")[0].strip()
                if lemma not in inverted_index: # Добавляем документ к множеству документов для данной леммы
                    inverted_index[lemma] = set()
                inverted_index[lemma].add(doc_id)

    # Сохраняем полученный индекс в файл
    with open(output_file, "w", encoding="utf-8") as f_out:
        for lemma, docs in sorted(inverted_index.items()):
            f_out.write(f"{lemma}: {' '.join(sorted(docs))}\n")

    print(f"Индекс сохранён в: {output_file}")
    return inverted_index


def load_inverted_index(index_file: str) -> dict[str, set[str]]:
    """
    Загружает инвертированный индекс из файла.
    """
    index = {}
    with open(index_file, "r", encoding="utf-8") as f:
        # Делим строку по первому ":", чтобы отделить лемму от списка документов
        for line in f:
            if ":" in line:
                lemma, docs = line.strip().split(":", 1)
                # Сохраняем лемму и множество ID документов (разделены пробелами)
                index[lemma] = set(docs.strip().split())
    return index


def lemmatize_query_token(token: str) -> str:
    """
       Преобразует токен в его лемматизированную (нормальную) форму.
    """
    return morph.parse(token)[0].normal_form


def preprocess_query(query: str) -> list[str]:
    """
    Преобразует запрос в список токенов (с лемматизацией терминов).
    """
    # Токенизация: слова, логические операторы и скобки
    raw_tokens = nltk.regexp_tokenize(query.lower(), pattern=r"[А-Яа-яЁё]+|AND|OR|NOT|\(|\)", gaps=False)

    processed = []
    for token in raw_tokens:
        if token.upper() in {"AND", "OR", "NOT"} or token in {"(", ")"}:
            # Сохраняем операторы и скобки
            processed.append(token.upper())
        else:
            # Лемматизируем все обычные слова
            lemma = lemmatize_query_token(token)
            processed.append(lemma)
    return processed


def infix_to_postfix(tokens: list[str]) -> list[str]:
    """
    Преобразует инфиксное выражение в постфиксное (Reverse Polish Notation).
    """
    precedence = {"NOT": 3, "AND": 2, "OR": 1} # Приоритеты операторов
    output = [] # Финальный постфиксный результат
    stack = [] # Стек для операторов и скобок

    for token in tokens:
        if token == "(":  # Скобка — просто кладем в стек
            stack.append(token)
        elif token == ")": # Закрывающая скобка — достаем всё до открывающей
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.pop() # удаляем открывающую скобку
        elif token in precedence: # Оператор: сравниваем приоритеты
            while stack and stack[-1] in precedence and precedence[token] <= precedence[stack[-1]]:
                output.append(stack.pop())
            stack.append(token)
        else: # Обычный термин — сразу в выходной список
            output.append(token)

    # Добавляем оставшиеся операторы из стека
    while stack:
        output.append(stack.pop())

    return output


def evaluate_postfix(postfix: list[str], index: dict[str, set[str]], all_docs: set[str]):
    """
    Выполняет булеву операцию над постфиксным выражением.
    """
    stack = []

    for token in postfix:
        if token == "AND":
            right = stack.pop() # второй операнд
            left = stack.pop() # первый операнд
            stack.append(left & right) # логическое И: пересечение множеств
        elif token == "OR":
            right = stack.pop()
            left = stack.pop()
            stack.append(left | right) # логическое ИЛИ: объединение
        elif token == "NOT":
            operand = stack.pop()
            stack.append(all_docs - operand) # логическое НЕ: всё, что не в operand
        else:
            stack.append(index.get(token, set())) # обычное слово → подставляем список документов

    return stack.pop() if stack else set()


def run_boolean_search(index_file: str, lemmas_dir: str, input_file:str):
    # Шаг 1: Загрузим или создадим индекс
    if not os.path.exists(index_file):
        print("Строим инвертированный индекс...")
        build_inverted_index(lemmas_dir=lemmas_dir, input_file=input_file, output_file=index_file)

    index = load_inverted_index(index_file)
    # извлекаем все уникальные идентификаторы документов, чтобы использовать их в логике NOT
    all_documents = {doc for docs in index.values() for doc in docs}

    # Шаг 2: Запрос пользователя
    print("\nВведите булев запрос (поддерживается AND, OR, NOT, скобки):")
    query = input(">> ")

    # Шаг 3: Обработка и вычисление
    tokens = preprocess_query(query)        # токенизация + лемматизация
    postfix = infix_to_postfix(tokens)      # перевод в постфиксную запись
    result_docs = evaluate_postfix(postfix, index, all_documents)   # вычисление результата

    # Шаг 4: Вывод
    print("\nДокументы, соответствующие запросу:")
    if result_docs:
        for doc in sorted(result_docs):
            print(f"  - {doc}")
    else:
        print("  Нет совпадений.")


if __name__ == "__main__":
    # Запуск функции булевого поиска с указанием:
    # - файла с инвертированным индексом "inverted_index.txt"
    # - директории с лемматизированными запросами "lemmas"
    # - шаблона имени файла с леммами "lemmas_*.txt"
    run_boolean_search(index_file="inverted_index.txt", lemmas_dir="lemmas", input_file="lemmas_*.txt")
