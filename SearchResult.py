class SearchResult:
    def __init__(self, web_page_name: str, web_page_url: str,
                 saved_html_name: str, similarity: float,
                 page_rank: float, score: float):
        self.web_page_name = web_page_name
        self.web_page_url = web_page_url
        self.saved_html_name = saved_html_name
        self.similarity = similarity
        self.page_rank = page_rank
        self.score = score