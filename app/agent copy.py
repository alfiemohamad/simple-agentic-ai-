import os
import vertexai
from google.oauth2 import service_account
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools.base import BaseTool
from langchain_google_vertexai.chat_models import ChatVertexAI

from app.config import settings
from app.memory import Neo4jMemory
from app.tools.weather import get_current_weather, get_forecast

# Initialize GCP Vertex AI (ensure GOOGLE_CLOUD_PROJECT is set)
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not credentials_path or not os.path.exists(credentials_path):
    raise EnvironmentError("Missing or invalid GOOGLE_APPLICATION_CREDENTIALS path.")

credentials = service_account.Credentials.from_service_account_file(
    credentials_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
vertexai.init(project=credentials.project_id, credentials=credentials, location="global")

# Short-term & long-term memory
short_term_mem = ConversationBufferMemory(memory_key="chat_history")
long_term_mem = Neo4jMemory(
    settings.NEO4J_URI,
    settings.NEO4J_USER,
    settings.NEO4J_PASSWORD,
    k=5
)

# Define weather tools
class CurrentWeatherTool(BaseTool):
    name = "get_current_weather"
    description = "Get current weather for a city"

    async def _arun(self, city: str) -> str:
        weather = await get_current_weather(city)
        return (
            f"Current weather in {weather['city']}, {weather['country']}\n"
            f"Temp: {weather['temperature']}°C (feels like {weather['feels_like']}°C)\n"
            f"Condition: {weather['condition'].title()}\n"
            f"Humidity: {weather['humidity']}%\n"
            f"Pressure: {weather['pressure']} hPa\n"
            f"Wind: {weather['wind_speed']} m/s\n"
            f"Visibility: {weather['visibility_km']} km"
        )

    def _run(self, city: str) -> str:
        raise NotImplementedError("Sync mode not supported. Use async.")

class ForecastTool(BaseTool):
    name = "get_weather_forecast"
    description = "Get 5-day weather forecast for a city"

    async def _arun(self, city: str) -> str:
        data = await get_forecast(city)
        lines = [f"5-day forecast for {data['city']}, {data['country']}:\n"]
        for f in data['forecast']:
            lines.append(
                f"• {f['datetime']}: {f['temperature']}°C, {f['condition'].title()}, Humidity: {f['humidity']}%"
            )
        return "\n".join(lines)

    def _run(self, city: str) -> str:
        raise NotImplementedError("Sync mode not supported. Use async.")

# Assemble tools
tools = [CurrentWeatherTool(), ForecastTool()]

# Create agent with VertexAI Gemini
llm = ChatVertexAI(
    model="gemini-2.0-flash-001",
    temperature=settings.AGENT_TEMPERATURE,
    max_tokens=None,
    max_retries=6
    
)
agent = initialize_agent(
    tools,
    llm,
    agent="chat-zero-shot-react-description",
    memory=short_term_mem,
    verbose=False,
)