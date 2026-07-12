"""
[src/agent.py]

교재 코드의 변수명/구조를 그대로 따름 (builder, agent, tool_node, graph).
"""
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from llm import build_llm  # [변경] generator.py -> llm.py로 이동됨
from tools import search_it_wiki, search_realtime_news

# === LLM과 도구 바인딩 === [교재]
tools = [search_it_wiki, search_realtime_news]
llm = build_llm()
llm_with_tools = llm.bind_tools(tools)


# === agent 노드: LLM 호출 === [교재]
def agent(state: MessagesState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# === ToolNode 생성 === [교재]
tool_node = ToolNode(tools=tools)

# === 그래프 빌더 === [교재]
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)

# === 엣지 구성 === [교재]
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)
builder.add_edge("tools", "agent")

# === 컴파일 === [교재]
memory = MemorySaver()  # 대화 기억용 보관함
graph = builder.compile(checkpointer=memory)


if __name__ == "__main__":
    test_questions = [
        "인공지능이 뭐야?",
        "오늘 삼성전자 관련 뉴스 알려줘",
    ]

    for q in test_questions:
        print(f"\n{'=' * 50}")
        print(f"질문: {q}")
        config = {"configurable": {"thread_id": "test-thread"}}
        result = graph.invoke({"messages": [HumanMessage(content=q)]}, config=config)

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [tc["name"] for tc in msg.tool_calls]
                print(f"  -> 호출된 도구: {tool_names}")

        print(f"답변: {result['messages'][-1].content}")