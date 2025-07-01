import aiohttp
from aiohttp import ClientSession, ClientTimeout
from typing import Dict, Any
from app.config import settings
from app.utils.error import WeatherAPIError

BASE_URL = "https://api.openweathermap.org/data/2.5"

async def fetch_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    timeout = ClientTimeout(total=30)
    async with ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise WeatherAPIError(f"{url} returned {resp.status}: {text}")
            return await resp.json()

async def get_current_weather(city: str) -> Dict[str, Any]:
    if not settings.OWM_API_KEY:
        raise WeatherAPIError("Missing OWM_API_KEY environment variable")
    data = await fetch_json(
        f"{BASE_URL}/weather",
        {"q": city, "units": "metric", "appid": settings.OWM_API_KEY},
    )
    return {
        "city": data["name"],
        "country": data["sys"]["country"],
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "condition": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data.get("wind", {}).get("speed", 0),
        "visibility_km": data.get("visibility", 0) / 1000,
    }

async def get_forecast(city: str) -> Dict[str, Any]:
    if not settings.OWM_API_KEY:
        raise WeatherAPIError("Missing OWM_API_KEY environment variable")
    data = await fetch_json(
        f"{BASE_URL}/forecast",
        {"q": city, "units": "metric", "appid": settings.OWM_API_KEY},
    )
    forecast_list = []
    for item in data.get("list", [])[:20]:
        forecast_list.append({
            "datetime": item.get("dt_txt"),
            "temperature": item["main"].get("temp"),
            "condition": item["weather"][0].get("description"),
            "humidity": item["main"].get("humidity"),
            "wind_speed": item.get("wind", {}).get("speed", 0),
        })
    return {"city": data["city"]["name"], "country": data["city"]["country"], "forecast": forecast_list}


