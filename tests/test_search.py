"""
[tests/test_search.py]

IT 인덱스 검색 품질 테스트. Gemini 호출 없음 (검색만).

실행: 저장소 루트에서 python3 tests/test_search.py
"""
import sys
import os

# src/ 폴더의 retriever.py를 import하기 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from retriever import load_hybrid_retriever

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FAISS_PATH = os.path.join(_DATA_DIR, "faiss_index_it")

retriever = load_hybrid_retriever(FAISS_PATH)

test_questions = [
    "인공지능이란 무엇인가?",
    "클라우드 컴퓨팅의 장점은?",
    "블록체인은 어떻게 작동하는가?",
    "HTTP 프로토콜이란?",
    "딥러닝과 머신러닝의 차이는?",
    "오늘 점심 뭐 먹지?",
    "이순신 장군의 업적은?",
]

for q in test_questions:
    results = retriever.invoke(q)
    print(f"\n질문: {q}")
    print(f"검색된 문서 수: {len(results)}개")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content[:80]}...")