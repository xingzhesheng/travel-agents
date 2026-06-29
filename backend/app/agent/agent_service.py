from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.config import config
from app.memory import memory_service
from app.tools.attractions_tool import attractions_tool
from app.tools.budget_tool import budget_tool
from app.tools.currency_tool import currency_tool
from app.tools.itinerary_tool import itinerary_tool
from app.tools.packing_tool import packing_tool
from app.tools.translator_tool import translator_tool
from app.tools.visa_tool import visa_tool
from app.tools.weather_tool import weather_tool

SYSTEM_PROMPT = """你是「旅途」AI旅行规划师，专业、热情的旅行助手。
不要输出思考过程，不要输出推理步骤，不要解释过程，直接给出最终答案。

可用工具：
- get_weather：查询目的地天气和穿衣建议
- get_attractions：推荐景点和目的地信息
- generate_itinerary：生成逐日详细行程
- calculate_budget：估算旅行总费用
- check_visa：查询签证要求
- convert_currency：货币换算和消费参考
- generate_packing_list：生成打包清单
- translate_phrases：旅行常用短语翻译

工作原则：
1. 根据用户需求主动使用合适工具，不要等用户指定
2. 复杂问题串联多个工具（如：查景点→生成行程→估算预算）
3. 回答热情友好，不使用 emoji
4. 使用中文回答
5. 输出的内容不要ai化，要像旅行达人一样分享经验和建议"""


class AgentService:
    def __init__(self) -> None:
        self.tools = [
            weather_tool, attractions_tool, itinerary_tool,
            budget_tool, visa_tool, currency_tool,
            packing_tool, translator_tool,
        ]

        if config.llm["provider"] == "deepseek":
            llm = ChatOpenAI(
                model=config.llm["model"],
                api_key=config.llm["api_key"],
                base_url=config.llm["base_url"],
                temperature=config.agent.temperature,
                streaming=True,
                max_tokens=512,
            )
        else:
            llm = ChatOllama(
                model=config.llm["model"],
                base_url=config.llm["base_url"],
                temperature=config.llm["temperature"],
                num_predict=512,
            )

        llm_with_tools = llm.bind_tools(self.tools)
        tool_node = ToolNode(self.tools)

        def should_continue(state: MessagesState):
            last = state["messages"][-1]
            return "tools" if getattr(last, "tool_calls", None) else END

        async def call_model(state: MessagesState):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        self.graph = graph.compile()

        print(f"✅ LangGraph Agent 已初始化，工具数：{len(self.tools)}")

    async def stream_chat(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        history = memory_service.get_history(user_id)
        messages = [*history, HumanMessage(content=message)]
        full_content = ""

        try:
            async for msg, meta in self.graph.astream(
                {"messages": messages},
                config={"recursion_limit": config.agent.max_iterations * 2},
                stream_mode="messages",
            ):
                if msg.content and meta.get("langgraph_node") == "agent":
                    chunk = msg.content if isinstance(msg.content, str) else ""
                    if chunk:
                        full_content += chunk
                        yield chunk

            memory_service.add_message(user_id, HumanMessage(content=message))
            memory_service.add_message(user_id, AIMessage(content=full_content))
        except Exception as error:
            yield f"\n\n抱歉，处理请求时出错：{error}"

    async def chat(self, user_id: str, message: str) -> str:
        history = memory_service.get_history(user_id)
        result = await self.graph.ainvoke(
            {"messages": [*history, HumanMessage(content=message)]},
            config={"recursion_limit": config.agent.max_iterations * 2},
        )
        last_ai = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        content = str(last_ai.content) if last_ai and last_ai.content else "抱歉，无法处理您的请求"
        memory_service.add_message(user_id, HumanMessage(content=message))
        memory_service.add_message(user_id, AIMessage(content=content))
        return content


agent_service = AgentService()
