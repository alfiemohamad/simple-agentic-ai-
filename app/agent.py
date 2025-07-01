# =======================
# app/agent.py (FIXED VERSION)
# =======================
from typing import Dict, Any
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

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
    num_ctx=2048
)

# Create a proper ReAct prompt template
prompt_template = PromptTemplate.from_template(
    """You are a helpful assistant that can check weather information.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)

tools = [CurrentWeatherTool(), ForecastTool()]

# Create the agent using the proper method
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
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
            
            # Simply pass the input as a dictionary with 'input' key
            result = await self.executor.ainvoke({
                "input": input_text
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