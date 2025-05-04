# для запросов
import requests
# для парсинга html
# pip install requests beautifulsoup4
from bs4 import BeautifulSoup
# для таймера
import time

# Парсит HTML и возвращает список найденных ссылок
def parse_links(html_text):

    soup = BeautifulSoup(html_text, 'html.parser')
    links = []

    # поиск по rel="category tag">Исследования
    # требуется найти <header class="entry-header">
	# 			<h2 class="entry-title"><a href="искомый URL"

    for cat_links in soup.find_all('div', class_='cat-links'):
        if 'Исследования' in cat_links.get_text():
            header = cat_links.find_next('header', class_='entry-header')
            if header:
                link = header.find('a', href=True)
                if link:
                    links.append(link['href'])
    return links

# Обработка успешного ответа: сохранение в файл и возврат результата парсинга
def parse_success_response(response, page_num, name_file):

    filename = f"{name_file}_{page_num}.html"

    # Сохранение HTML в файл
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response.text)

    # Парсинг и сбор результатов
    return parse_links(response.text)

def get_links(url, page_num, name_file, all_links):
    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            page_links = parse_success_response(response, page_num, name_file)
            all_links.extend(page_links)
            print(f"Найдено ссылок на странице: {len(page_links)}")
        else:
            print(f"Ошибка {response.status_code} для страницы {page_num}")

        # Задержка между запросами, чтобы избежать блокировки
        time.sleep(2)

    except Exception as e:
        print(f"Ошибка при обработке страницы {page_num}: {str(e)}")

def process_success_response(response, page_num, name_file):

    filename = f"links/{name_file}_{page_num}.html"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(response.text)

def get_response_links(url, page_num, name_file):
    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            process_success_response(response, page_num, name_file)

        else:
            print("Возникла ошибка")

        # Задержка между запросами, чтобы избежать блокировки
        time.sleep(2)

    except Exception as e:
        print(f"Ошибка при обработке страницы {page_num}: {str(e)}")

def main():
    base_url = "https://www.trv-science.ru/category/news/"
    total_pages = 2
    all_links = []

    for page_num in range(1, total_pages + 1):

        if page_num == 1:
            url = base_url
        else:
            url = f"{base_url}page/{page_num}/"

        print(f"\nОбрабатываю страницу {page_num}...")

        get_links(url, page_num, "page", all_links)

    # Фильтрация уникальных ссылок с сохранением порядка
    unique_links = []
    seen = set()
    for link in all_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    # Запись первых 100 уникальных ссылок в файл
    with open('index.txt', 'w', encoding='utf-8') as f:
        for i, link in enumerate(unique_links[:100], 1):
            f.write(f"unloading_{i} - {link}\n")

    print("\n" + "=" * 50)
    print(f"Всего найдено ссылок: {len(all_links)}")
    print(f"Уникальных ссылок: {len(unique_links)}")
    print(f"Первые 100 уникальных ссылок сохранены в index.txt")
    print("\nСписок уникальных ссылок:")

    # вывод всех найденных ссылок на сайте https://www.trv-science.ru/category/news/ в рубрике: Исследования
    for i, link in enumerate(unique_links, 1):
        print(f"{i}. {link}")

    # выкачка html кода всех найденных статей по ранее найденным ссылкам
    for i, link in enumerate(unique_links[:100], 1):
        get_response_links(link, i, "unloading")
    print(f"Первые 100 уникальных ссылок сохранены в links/unloading_1-100.txt")

if __name__ == "__main__":
    main()