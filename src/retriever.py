"""
retriever.py - 하이브리드 검색기 (FAISS + BM25/Mecab) + Reranking

검색 흐름:
  1차: FAISS + BM25 하이브리드로 후보 10개를 넉넉하게 뽑음
  2차: Cross-encoder(bge-reranker-v2-m3)로 "질문-문서" 쌍을 재채점
  3차: 재채점 상위 3개만 최종 반환

리랭킹을 추가한 이유:
  평가 결과 context_precision이 0.33으로 낮았음 (검색된 3개 중 1개만 관련 있었음).
  1차 검색에서 인접 분야 문서가 섞여 들어오는 문제를 2차 재채점으로 걸러내기 위함.
"""
import logging
import warnings

logging.getLogger("langchain").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab
from sentence_transformers import CrossEncoder

MECAB_DICPATH = "/opt/homebrew/lib/mecab/dic/mecab-ko-dic"


def load_hybrid_retriever(
    faiss_path,
    embedding_model="BAAI/bge-m3",
    reranker_model="BAAI/bge-reranker-v2-m3",
    k=3,
    initial_k=10,
    device="cpu",
    score_threshold=0.5,
    alpha=0.5,
    bm25_min_score=12.0,
    relaxed_score_threshold=0.3,
    relaxed_bm25_min_score=6.0,
    use_reranker=True,
):
    """
    FAISS + BM25 하이브리드 검색 + Cross-encoder 리랭킹.

    use_reranker=True:  1차로 initial_k(10)개 뽑고, 리랭커로 재채점 후 상위 k(3)개 반환
    use_reranker=False: 기존 방식 그대로 (리랭킹 전 베이스라인과 비교할 때 사용)
    """
    print("임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("FAISS 벡터 저장소 로드 중...")
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    all_docs = list(vectorstore.docstore._dict.values())
    for i, doc in enumerate(all_docs):
        doc.metadata["_idx"] = i
    print(f"문서 {len(all_docs)}개 로드 완료")

    mecab = Mecab(dicpath=MECAB_DICPATH)

    print("BM25 인덱스 구축 중 (Mecab으로 명사 추출)...")
    tokenized_corpus = [mecab.nouns(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 인덱스 구축 완료")

    reranker = None
    if use_reranker:
        print(f"리랭커 로드 중... ({reranker_model})")
        reranker = CrossEncoder(reranker_model)
        print("리랭커 로드 완료")

    def _search_once(question, vec_threshold, bm25_threshold, num_results):
        vec_results = vectorstore.similarity_search_with_relevance_scores(
            question, k=len(all_docs)
        )
        vec_scores = {
            doc.metadata["_idx"]: score
            for doc, score in vec_results
            if score >= vec_threshold
        }

        tokenized_query = mecab.nouns(question)
        bm25_scores_raw = bm25.get_scores(tokenized_query) if tokenized_query else []

        relevant_raw = [s for s in bm25_scores_raw if s >= bm25_threshold]
        max_bm25 = max(relevant_raw) if relevant_raw else 1
        bm25_scores = {
            i: bm25_scores_raw[i] / max_bm25
            for i in range(len(bm25_scores_raw))
            if bm25_scores_raw[i] >= bm25_threshold
        }

        candidate_ids = set(vec_scores) | set(bm25_scores)
        combined = {
            idx: alpha * vec_scores.get(idx, 0) + (1 - alpha) * bm25_scores.get(idx, 0)
            for idx in candidate_ids
        }

        top_ids = sorted(combined, key=combined.get, reverse=True)[:num_results]
        return [all_docs[i] for i in top_ids]

    def _rerank(question, docs):
        if not docs:
            return []
        pairs = [[question, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        filtered = [(doc, score) for doc, score in ranked if score > 0.01]
        return [doc for doc, score in filtered[:k]]

    def retrieve(question):
        search_k = initial_k if use_reranker else k

        results = _search_once(question, score_threshold, bm25_min_score, search_k)

        if not results:
            results = _search_once(question, relaxed_score_threshold, relaxed_bm25_min_score, search_k)

        if use_reranker and results:
            results = _rerank(question, results)

        return results

    return RunnableLambda(retrieve)


load_retriever = load_hybrid_retriever


if __name__ == "__main__":
    import sys
    import os
    _DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    FAISS_PATH = os.path.join(_DATA_DIR, "faiss_index_it")

    retriever = load_hybrid_retriever(FAISS_PATH)

    test_questions = [
        "인공지능이 뭐야?",
        "딥러닝과 머신러닝의 차이는?",
        "클라우드 컴퓨팅의 장점은?",
        "오늘 점심 뭐 먹지?",
    ]
    for q in test_questions:
        results = retriever.invoke(q)
        print(f"\n질문: {q}")
        print(f"검색된 문서 수: {len(results)}개")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.page_content[:100]}...")
