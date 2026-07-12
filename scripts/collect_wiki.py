"""
[scripts/collect_wiki.py]

한국어 위키백과에서 IT 관련 문서를 수집하는 스크립트.
결과는 저장소 루트의 data/ 폴더에 저장한다.

실행: 저장소 루트에서 python3 scripts/collect_wiki.py
"""
import json
import os
import time
import wikipediaapi

SEED_TERMS = [
    "알고리즘", "자료 구조", "데이터베이스", "운영 체제", "컴퓨터 네트워크",
    "API", "프로그래밍 언어", "오픈 소스",
    "인공지능", "기계 학습", "딥 러닝", "자연어 처리", "빅 데이터", "데이터 마이닝",
    "클라우드 컴퓨팅", "서버", "HTTP", "웹 브라우저", "도메인 네임 시스템",
    "정보 보안", "암호화", "해킹", "방화벽",
    "블록체인", "사물인터넷", "가상 현실", "5G", "반도체",
    "스타트업", "전자 상거래",
]

MAX_LINKS_PER_PAGE = 15
MIN_TEXT_LENGTH = 300

# data/ 폴더 경로 (이 파일 위치 기준: scripts/ -> 저장소 루트 -> data/)
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(_DATA_DIR, "it_wiki_docs.json")

EXCLUDE_KEYWORDS = ["목록", "연표", "분류:", "틀:", "위키", "동음이의"]

wiki = wikipediaapi.Wikipedia(
    user_agent="rain-research-bot (KTB4 bootcamp project)",
    language="ko",
)


def is_valid_title(title: str) -> bool:
    if any(kw in title for kw in EXCLUDE_KEYWORDS):
        return False
    if title.strip().isdigit():
        return False
    return True


def fetch_page(title: str):
    page = wiki.page(title)
    time.sleep(0.1)
    if not page.exists():
        return None
    if len(page.text) < MIN_TEXT_LENGTH:
        return None
    return page


def main():
    os.makedirs(_DATA_DIR, exist_ok=True)  # data/ 폴더 없으면 생성

    collected = {}
    failed = []

    print(f"시드 {len(SEED_TERMS)}개 수집 시작...")
    seed_pages = []
    for term in SEED_TERMS:
        page = fetch_page(term)
        if page is None:
            failed.append(term)
            print(f"  [실패] {term} (문서 없음 또는 너무 짧음)")
            continue
        collected[page.title] = page.text
        seed_pages.append(page)
        print(f"  [수집] {page.title} ({len(page.text)}자)")

    print(f"\n링크 확장 수집 시작 (문서당 최대 {MAX_LINKS_PER_PAGE}개)...")
    for page in seed_pages:
        links = [t for t in page.links.keys() if is_valid_title(t)]
        for link_title in links[:MAX_LINKS_PER_PAGE]:
            if link_title in collected:
                continue
            linked = fetch_page(link_title)
            if linked is None:
                continue
            collected[linked.title] = linked.text
        print(f"  [확장 완료] {page.title} -> 누적 {len(collected)}개")

    docs = [{"title": t, "text": x} for t, x in collected.items()]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(docs)}개 문서 저장 완료 -> {OUTPUT_FILE}")
    if failed:
        print(f"시드 중 실패: {failed}")


if __name__ == "__main__":
    main()