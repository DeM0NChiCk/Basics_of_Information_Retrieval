import os
import re
import glob
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import pymorphy2

# Если запускаете в первый раз, раскомментируйте эти строки (нужен разовый запуск):
# nltk.download('punkt')
# nltk.download('stopwords')

# Инициализируем морфологический анализатор и стоп-слова
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

# Расширение для списка стоп-слов
extra_stops = {"что", "это", "так", "вот", "быть", "как", "к", "—", "–", "аж", "тыс", "млн", "млрд"}
russian_stopwords |= extra_stops

def extract_visible_text_from_html(html_content: str) -> str:
    """
    Извлекает из HTML только видимый текст, вырезая скрипты и стили.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Находим блок текста с id="main"
    main_div = soup.find('main', id='main')
    if not main_div:
        return ""  # Если такого элемента нет, возвращаем пустую строку

    # Находим первый <div> внутри main_div (эквивалент XPath: //*[@id="main"]/div[1])
    target_divs = main_div.find_all('div', recursive=False)
    if not target_divs:
        return ""  # Если внутри нет вложенных <div>, то возвращаем пустую строку
    target_div = target_divs[0]  # первый div внутри <div id="main">

    # Удаляем теги <script> и <style> из target_div
    for tag in target_div(["script", "style"]):
        tag.decompose()

    # Извлекаем текст
    text = target_div.get_text(separator=" ")
    return text

def tokenize_text(text: str) -> list:
    """
    Делит текст на токены (слова), удаляя "мусор" и стоп-слова.
    Возвращает список уникальных токенов.
    """
    # Приведём текст к нижнему регистру
    text = text.lower()

    # Разбиваем на токены с помощью nltk
    raw_tokens = nltk.regexp_tokenize(text, pattern=r"[А-Яа-яЁё]+", gaps=False)

    unique_tokens = set()
    for token in raw_tokens:
        # Пропускаем стоп-слова, а также любые пустые строки
        if token in russian_stopwords or not token.strip():
            continue

        # Отбрасываем токены, содержащие цифры (числа или смешанные слова)
        if re.search(r"\d", token):
            continue

        # Убираем слишком короткие слова (длина меньше 2 символов)
        if len(token) < 2:
            continue

        unique_tokens.add(token)

    return sorted(unique_tokens)

def lemmatize_tokens(tokens: list) -> dict:
    """
    Лемматизирует список токенов и возвращает словарь вида:
    {лемма: {токен1, токен2, ...}, ...}
    """
    lemma_dict = {}
    for token in tokens:
        # Получаем лемму через pymorphy2
        p = morph.parse(token)[0]
        lemma = p.normal_form

        if lemma not in lemma_dict:
            lemma_dict[lemma] = set()
        lemma_dict[lemma].add(token)

    return lemma_dict

def process_html_files(input_folder: str):
    """
    Ищет в папке `input_folder` файлы `page_*.html`,
    для каждого файла извлекает токены, леммы и записывает в текстовые файлы.
    """
    # Ищем все HTML-файлы вида page_*.html
    html_files = glob.glob(os.path.join(input_folder, "unloading_*.html"))

    for html_file in html_files:
        # Читаем содержимое
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Извлекаем видимый текст
        visible_text = extract_visible_text_from_html(html_content)

        # Токенизируем и фильтруем
        tokens = tokenize_text(visible_text)

        # Лемматизируем и группируем
        lemma_dict = lemmatize_tokens(tokens)

        # Формируем имена выходных файлов
        base_name = os.path.splitext(os.path.basename(html_file))[0]  # page_N
        tokens_output_file = f"tokens_{base_name}.txt"
        lemmas_output_file = f"lemmas_{base_name}.txt"

        os.makedirs("tokens", exist_ok=True)
        # Запись токенов в файл
        with open(os.path.join("tokens", tokens_output_file), "w", encoding="utf-8") as f_tokens:
            for token in tokens:
                f_tokens.write(token + "\n")

        os.makedirs("lemmas", exist_ok=True)
        # Запись лемм
        with open(os.path.join("lemmas", lemmas_output_file), "w", encoding="utf-8") as f_lemmas:
            for lemma, token_set in sorted(lemma_dict.items()):
                # Пример формата: "год: год года годами"
                line = lemma + ": " + " ".join(sorted(token_set)) + "\n"
                f_lemmas.write(line)

        print(f"Обработан файл: {html_file}")
        print(f"  Токены  -> {tokens_output_file}")
        print(f"  Леммы   -> {lemmas_output_file}\n")


if __name__ == "__main__":
    # Запускаем процесс для папки links
    process_html_files(input_folder="links")
