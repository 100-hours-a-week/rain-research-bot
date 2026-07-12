"""
[naver_news_tool.py]

네이버 뉴스 검색 API로 실시간 뉴스를 가져오는 도구.
IT 위키 인덱스(고정 지식)와 짝을 이루는, "매번 바뀌는 정보" 담당 도구.

실행 전: .env에 NAVER_CLIENT_ID, NAVER_CLIENT_SECRET 있어야 함
테스트: python3 naver_news_tool.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"


def search_news(query: str, display: int = 5) -> list[dict]:
    """
    네이버 뉴스 검색.
    Returns: [{"title": str, "description": str, "link": str, "pubDate": str}, ...]
    (title/description은 <b> 태그가 섞여있어 아래에서 제거함)
    """
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "sort": "date"}  # 최신순 정렬

    response = requests.get(NAVER_NEWS_URL, headers=headers, params=params)
    response.raise_for_status()
    items = response.json().get("items", [])

    results = []
    for item in items:
        results.append({
            "title": _strip_tags(item["title"]),
            "description": _strip_tags(item["description"]),
            "link": item["link"],
            "pubDate": item["pubDate"],
        })
    return results


def _strip_tags(text: str) -> str:
    """네이버 API 응답에 섞인 <b>강조</b> 태그와 특수문자 제거"""
    return text.replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&")


if __name__ == "__main__":
    results = search_news("삼성전자")
    print(f"검색된 뉴스 {len(results)}개\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['title']}")
        print(f"    {r['description'][:80]}...")
        print(f"    {r['pubDate']}")
        print(f"    링크: {r['link']}\n")