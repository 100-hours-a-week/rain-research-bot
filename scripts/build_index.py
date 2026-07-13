"""
[scripts/build_index.py]

data/it_wiki_docs_clean.json을 읽어서 청킹 -> 임베딩 -> FAISS 인덱스로 저장.
결과는 data/faiss_index_it/ 에 생성된다.

실행: 저장소 루트에서 python3 scripts/build_index.py
"""
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_FILE = os.path.join(_DATA_DIR, "it_wiki_docs_clean.json")
OUTPUT_DIR = os.path.join(_DATA_DIR, "faiss_index_it")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        docs_raw = json.load(f)
    print(f"문서 {len(docs_raw)}개 로드")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for doc in docs_raw:
        title, text = doc["title"], doc["text"]
        for piece in splitter.split_text(text):
            chunks.append(
                Document(
                    page_content=f"[{title}] {piece}",
                    metadata={"title": title},
                )
            )
    print(f"청크 {len(chunks)}개 생성 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print("임베딩 모델 로드 중... (bge-m3)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("FAISS 인덱스 구축 중...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(OUTPUT_DIR)
    print(f"저장 완료 -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()