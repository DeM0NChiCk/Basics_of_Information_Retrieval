from typing import List
from flask import Flask, request, render_template_string, redirect, url_for
from vector_search import search as vector_search
from SearchResult import SearchResult


def do_search(query: str, mode: str) -> List[SearchResult]:
    results = vector_search(query, mode)
    return results[:10]


# -----------------------------------------------------------------------------

app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<title>Поисково-локальный поиск</title>
<meta charset="utf-8">
<style>
  body      { font-family: system-ui, sans-serif; margin: 2rem 4rem; }
  form      { margin-bottom: 1.5rem; display: flex; gap: 1rem; align-items: center; }
  input[type=text] { flex: 1 0 15rem; padding: .4rem .6rem; font-size: 1rem; }
  label     { display: flex; align-items: center; gap: .25rem; }
  ul        { padding-left: 1.1rem; }
  li        { margin-bottom: .6rem; }
  .score    { display:block; margin-left:1rem; font-size: .85rem; color:#555; }
</style>

<h1>🔍 Поисково-локальный поиск</h1>

<form method="GET" action="{{ url_for('search') }}">
  <input name="q"  type="text" placeholder="Поиск…" value="{{ query|default('') }}" autofocus>
  <label><input type="radio" name="mode" value="lemmas"  {{ 'checked' if mode!='tokens' }}> леммы</label>
  <label><input type="radio" name="mode" value="tokens"  {{ 'checked' if mode=='tokens' }}> токены</label>
  <button type="submit">Найти</button>
</form>

{% if results is not none %}
  {% if results %}
    <p><strong>Найдено результатов</strong>:{{ results|length }}</p>
    <ul>
      {% for r in results %}
        <li>
          <a href="{{ r.web_page_url }}" target="_blank">{{ r.web_page_name }}</a>
          <span class="score">(схожесть: {{ "%.4f"|format(r.similarity) }}, файл: {{ r.saved_html_name }})</span>
        </li>
      {% endfor %}
    </ul>
  {% else %}
    <p>Ничего не нашлось 😔</p>
  {% endif %}
{% endif %}
"""


@app.get("/")
def home():
    return render_template_string(TEMPLATE, results=None, query="", mode="lemmas")


@app.get("/search")
def search():
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "lemmas")
    if not query:
        return redirect(url_for("home"))
    results = do_search(query, mode)
    return render_template_string(
        TEMPLATE,
        results=results,
        query=query,
        mode=mode
    )


if __name__ == "__main__":
    app.run(debug=True)
