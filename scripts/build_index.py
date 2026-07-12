"""
[build_index.py]

collect_wiki.py 가 만든 it_wiki_docs.json 을 읽어서
청킹 -> 임베딩 -> FAISS 인덱스로 저장하는 스크립트.

방식은 6주차(KorQuAD 인덱싱)와 동일:
  - chunk_size=500, overlap=50 (검증된 값 그대로)
  - BAAI/bge-m3 로컬 임베딩
차이점 하나:
  - 각 청크 맨 앞에 "[문서제목]"을 붙여넣음 (제목 컨텍스트 주입)
    -> 긴 문서가 잘리면서 뒷쪽 청크가 무슨 주제인지 잃어버리는 문제 방지
    -> IT 용어는 제목 매칭이 특히 중요하므로 검색 품질에 직접 영향

실행: python3 build_index.py
소요 시간: 문서 수에 따라 수십 분 (bge-m3 임베딩이 오래 걸림, CPU 기준)
결과: faiss_index_it/ 폴더 생성
"""

import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

INPUT_FILE = "it_wiki_docs_clean.json"
OUTPUT_DIR = "faiss_index_it"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"


def main():
    # ---- 1. 문서 로드 ----
    with open(INPUT_FILE, encoding="utf-8") as f:
        docs_raw = json.load(f)
    print(f"문서 {len(docs_raw)}개 로드")

    # ---- 2. 청킹 (+ 제목 컨텍스트 주입) ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for doc in docs_raw:
        title, text = doc["title"], doc["text"]
        for piece in splitter.split_text(text):
            chunks.append(
                Document(
                    page_content=f"[{title}] {piece}",  # 제목을 청크 맨 앞에 주입
                    metadata={"title": title},
                )
            )
    print(f"청크 {len(chunks)}개 생성 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # ---- 3. 임베딩 + FAISS 저장 ----
    print("임베딩 모델 로드 중... (bge-m3)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("FAISS 인덱스 구축 중... (문서 수에 따라 수십 분 걸릴 수 있음)")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(OUTPUT_DIR)
    print(f"저장 완료 -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()