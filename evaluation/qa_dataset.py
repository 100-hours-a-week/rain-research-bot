# [evaluation/qa_dataset.py]
#
# 라우팅 정확도용: IT 질문 3개 + 뉴스 질문 3개 (Agent가 맞는 도구를 고르는지 확인)
# 답변 품질용: IT 질문 3개만 (뉴스는 매번 답이 바뀌어서 고정 정답을 둘 수 없음)

ROUTING_QUESTIONS = [
    {"question": "인공지능이 뭐야?", "expected_tool": "search_it_wiki"},
    {"question": "블록체인은 어떻게 작동해?", "expected_tool": "search_it_wiki"},
    {"question": "클라우드 컴퓨팅의 장점은?", "expected_tool": "search_it_wiki"},
    {"question": "오늘 삼성전자 뉴스 알려줘", "expected_tool": "search_realtime_news"},
    {"question": "SK하이닉스 최근 소식은?", "expected_tool": "search_realtime_news"},
    {"question": "카카오 관련 뉴스 있어?", "expected_tool": "search_realtime_news"},
]

# 답변 품질 평가용 (IT 개념 질문만 - 정답이 시간 지나도 안 바뀜)
QUALITY_QUESTIONS = [
    {
        "question": "인공지능이 뭐야?",
        "expected_answer": "인간의 학습·추론·지각 능력을 컴퓨터로 구현하는 기술",
        "expected_keywords": ["학습"],
    },
    {
        "question": "블록체인은 어떻게 작동해?",
        "expected_answer": "데이터를 블록 단위로 나눠 P2P 방식으로 분산 저장하는 기술",
        "expected_keywords": ["블록"],
    },
    {
        "question": "클라우드 컴퓨팅의 장점은?",
        "expected_answer": "초기 구입 비용이 적고 컴퓨터 가용율이 높음",
        "expected_keywords": ["비용"],
    },
]
