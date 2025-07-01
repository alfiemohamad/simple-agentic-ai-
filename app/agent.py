from typing import Dict, Any
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.memory import Neo4jMemory
from app.tools.weather import get_current_weather, get_forecast

# Memory setup
short_term_mem = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
long_term_mem = Neo4jMemory(
    settings.NEO4J_URI,
    settings.NEO4J_USER,
    settings.NEO4J_PASSWORD,
    k=5
)

# Weather Tools
class CurrentWeatherTool(BaseTool):
    name: str = "get_current_weather"
    description: str = "Get current weather for a city. Input should be the city name."
    
    async def _arun(self, city: str) -> str:
        weather = await get_current_weather(city)
        return (
            f"Weather in {weather['city']}, {weather['country']}:\n"
            f"- Temperature: {weather['temperature']}°C\n"
            f"- Condition: {weather['condition']}\n"
            f"- Humidity: {weather['humidity']}%\n"
            f"- Wind: {weather['wind_speed']} m/s"
        )

    def _run(self, city: str) -> str:
        raise NotImplementedError("Sync execution not supported")

class ForecastTool(BaseTool):
    name: str = "get_weather_forecast"
    description: str = "Get 5-day weather forecast for a city. Input should be the city name."
    
    async def _arun(self, city: str) -> str:
        data = await get_forecast(city)
        forecast = "\n".join(
            f"{f['datetime']}: {f['temperature']}°C, {f['condition']}"
            for f in data['forecast']
        )
        return f"5-day forecast for {data['city']}:\n{forecast}"

    def _run(self, city: str) -> str:
        raise NotImplementedError("Sync execution not supported")

# Initialize Ollama
llm = Ollama(
    model="llama3",
    temperature=0.7,
    top_p=0.9,
    num_ctx=2048,
    system="""You are a helpful assistant. Follow these rules:
1. Use tools when needed
2. Respond in this format:
   Action: tool_name
   Action Input: input
   OR
   Final Answer: response"""
)

# Agent setup
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant that can ONLY use these tools: get_current_weather, get_weather_forecast. "
     "When you need to use a tool, ALWAYS respond ONLY in this format:\n"
     "Action: <tool_name>\n"
     "Action Input: <input>\n"
     "Do not use any other tool name. Do not translate. Do not explain. Do not invent tools."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tools = [CurrentWeatherTool(), ForecastTool()]

agent = (
    prompt
    | llm.bind(stop=["Observation:"])
    | ReActSingleInputOutputParser()
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=short_term_mem,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

class AgentWrapper:
    def __init__(self):
        self.executor = agent_executor
        
    async def ainvoke(self, input_text: str) -> Dict[str, Any]:
        try:
            if not isinstance(input_text, str):
                raise ValueError("Input must be a string")
            
            result = await self.executor.ainvoke({
                "input": input_text,
                "chat_history": short_term_mem.load_memory_variables({})["chat_history"],
                "agent_scratchpad": [],  # <-- Tambahkan ini!
            })
            
            return {
                "output": result.get("output", "No response generated"),
                "intermediate_steps": result.get("intermediate_steps", [])
            }
        except Exception as e:
            return {
                "error": str(e),
                "output": f"Error processing request: {str(e)}"
            }