# [evaluation/metrics.py]
#
# 5개 지표: keyword_recall(직접 계산) + faithfulness/answer_relevancy/
# context_precision/context_recall(LLM 호출 1번으로 통합, RAGAS 표준 개념 참고해 구현)

import json
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from llm import build_llm

judge_llm = build_llm()


def _ask_judge(prompt: str) -> dict:
    response = judge_llm.invoke(prompt)
    time.sleep(3)
    text = response.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  [경고] 채점 결과 파싱 실패: {text[:100]}")
        return {}


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    hit = sum(1 for kw in expected_keywords if kw in answer)
    return hit / len(expected_keywords)


def judge_all(question: str, answer: str, context: str, expected_answer: str) -> dict:
    """faithfulness/answer_relevancy/context_precision/context_recall을 호출 1번으로 채점"""
    context_text = context if context else "(검색된 문서 없음)"

    prompt = f"""당신은 RAG 시스템 평가자입니다. 아래 정보를 보고 4가지 항목을 채점하세요.

[질문]
{question}

[정답]
{expected_answer}

[검색된 문서/뉴스]
{context_text}

[생성된 답변]
{answer}

채점 기준:
1. faithfulness: 답변이 검색된 내용에만 근거하는지 (0.0=지어냄, 1.0=근거함)
2. answer_relevancy: 답변이 질문에 직접적으로 관련된 내용인지 (0.0=무관, 1.0=정확히 답함)
3. context_precision: 검색된 내용이 질문과 관련 있는지 (0.0=무관, 1.0=관련 있음)
4. context_recall: 검색된 내용만으로 정답을 알아낼 수 있는지 (0.0=불가능, 1.0=가능)

다음 JSON 형식으로만 답하세요:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}}"""

    result = _ask_judge(prompt)
    return {
        "faithfulness": float(result.get("faithfulness", 0.0)),
        "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
        "context_precision": float(result.get("context_precision", 0.0)),
        "context_recall": float(result.get("context_recall", 0.0)),
    }
