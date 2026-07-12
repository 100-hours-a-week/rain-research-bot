"""
retriever.py - 하이브리드 검색기 (FAISS + BM25/Mecab), 직접 구현 버전

이전 버전은 LangChain의 EnsembleRetriever(라이브러리)로 FAISS와 BM25를
합쳤지만, 이 버전은 그 결합 로직을 직접 구현했다.

[rain-research-bot 전용 추가사항]
1차 검색(기본 threshold) 결과가 0개면, threshold를 완화해서 1번 더 시도한다.
(07의 원본 retriever.py에는 없던 로직 - IT 위키 인덱스에서 "인공지능"처럼
 threshold 경계선에 걸리는 질문이 발견되어 추가함)
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

MECAB_DICPATH = "/opt/homebrew/lib/mecab/dic/mecab-ko-dic"


def load_hybrid_retriever(
    faiss_path,
    embedding_model="BAAI/bge-m3",
    k=3,
    device="cpu",
    score_threshold=0.5,
    alpha=0.5,
    bm25_min_score=12.0,
    relaxed_score_threshold=0.3,   # [추가] 재시도용 완화 기준 (벡터)
    relaxed_bm25_min_score=6.0,    # [추가] 재시도용 완화 기준 (BM25)
):
    """
    FAISS(의미 기반) + BM25(Mecab 형태소 분석 기반 키워드 검색)를 직접
    결합한 하이브리드 retriever를 생성한다.

    1차 검색이 0개면, threshold를 완화해서 2차 검색을 자동으로 1번 더 시도한다.
    (그래도 0개면 최종적으로 빈 리스트 반환 - "진짜 관련 없다"는 뜻으로 봄)
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

    def _search_once(question: str, vec_threshold: float, bm25_threshold: float):
        """threshold 값을 받아서 검색 1회 수행 (내부 헬퍼 함수)"""
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

        top_ids = sorted(combined, key=combined.get, reverse=True)[:k]
        return [all_docs[i] for i in top_ids]

    def retrieve(question: str):
        # ---- 1차 검색: 기본(엄격한) threshold ----
        results = _search_once(question, score_threshold, bm25_min_score)

        # ---- [추가] 0개면, threshold 완화해서 2차 검색 1번 더 시도 ----
        if not results:
            results = _search_once(question, relaxed_score_threshold, relaxed_bm25_min_score)

        return results

    return RunnableLambda(retrieve)


load_retriever = load_hybrid_retriever


if __name__ == "__main__":
    retriever = load_hybrid_retriever("faiss_index_it")

    test_questions = [
        "인공지능이 뭐야?",
        "오늘 날씨에 대해 알려줘",
        "클라우드 컴퓨팅의 장점은?",
    ]
    for q in test_questions:
        results = retriever.invoke(q)
        print(f"\n질문: {q}")
        print(f"검색된 문서 수: {len(results)}개")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.page_content[:100]}...")