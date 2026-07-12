"""
generator.py - RAG 답변 생성기 (Generator), LangChain 버전

6주차에서는 google.generativeai를 직접 호출해서 답변을 생성했다.
7주차에서는 이를 LangChain의 ChatGoogleGenerativeAI + ChatPromptTemplate으로
교체해서, LCEL 파이프(`|`)로 다른 컴포넌트와 자연스럽게 연결할 수 있도록 했다.
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 불러오기


def build_llm(model_name="gemini-2.5-flash"):
    """
    LangChain에서 사용할 Gemini LLM 객체를 생성하는 함수

    Returns:
        llm: LangChain의 Runnable 인터페이스를 따르는 LLM 객체
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
    )
    return llm


def build_prompt():
    """
    검색된 문서(context)와 질문(question)을 받아 프롬프트를 구성하는
    ChatPromptTemplate을 생성하는 함수.

    6주차의 build_prompt()가 문자열을 직접 조합했다면, 여기서는
    LangChain이 제공하는 템플릿 객체를 사용해 "{context}", "{question}"
    변수를 자동으로 채워 넣도록 한다.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "다음 문서들을 참고해서 질문에 답변해주세요. "
         "문서에 없는 내용은 추측하지 말고, 모르면 모른다고 답변해주세요.\n\n"
         "[참고 문서]\n{context}"),
        ("human", "{question}"),
    ])
    return prompt


def format_docs(docs):
    """
    retriever가 반환한 문서 리스트를, 프롬프트에 넣기 좋은
    하나의 문자열로 합치는 함수.
    """
    return "\n\n".join(
        f"[문서 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )


if __name__ == "__main__":
    from langchain_core.output_parsers import StrOutputParser

    llm = build_llm()
    prompt = build_prompt()

    # 간단히 문서 없이 프롬프트 -> LLM 연결만 테스트
    test_chain = prompt | llm | StrOutputParser()
    answer = test_chain.invoke({
        "context": "이순신의 장인은 보성군수를 역임한 방진(方震)이다.",
        "question": "이순신의 장인은 누구인가?"
    })
    print(answer)