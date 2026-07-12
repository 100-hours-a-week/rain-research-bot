"""
[scripts/clean_wiki_docs.py]

data/it_wiki_docs.json의 각 문서에서 부록 섹션을 잘라내고,
data/it_wiki_docs_clean.json으로 저장한다.

실행: 저장소 루트에서 python3 scripts/clean_wiki_docs.py
"""
import json
import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_FILE = os.path.join(_DATA_DIR, "it_wiki_docs.json")
OUTPUT_FILE = os.path.join(_DATA_DIR, "it_wiki_docs_clean.json")

BOILERPLATE_MARKERS = [
    "같이 보기", "함께 보기", "각주", "외부 링크", "외부링크",
    "참고 문헌", "참고문헌", "주석", "메모", "참조", "소프트웨어",
]


def clean_text(text: str) -> str:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip() in BOILERPLATE_MARKERS:
            return "\n".join(lines[:i]).strip()
    return text.strip()


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        docs = json.load(f)

    cleaned = []
    total_removed_chars = 0
    for doc in docs:
        original_len = len(doc["text"])
        new_text = clean_text(doc["text"])
        total_removed_chars += original_len - len(new_text)
        cleaned.append({"title": doc["title"], "text": new_text})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    avg_removed = total_removed_chars / len(docs)
    print(f"문서 {len(docs)}개 정제 완료 -> {OUTPUT_FILE}")
    print(f"문서당 평균 {avg_removed:.0f}자 제거됨 (부록 섹션)")


if __name__ == "__main__":
    main()