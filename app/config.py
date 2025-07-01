import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OWM_API_KEY: str = os.getenv("OWM_API_KEY")
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USER: str = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", 0))

settings = Settings()



