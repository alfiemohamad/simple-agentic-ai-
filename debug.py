import asyncio
from app.tools.weather import get_current_weather

async def main():
    city = "Jakarta"
    try:
        result = await get_current_weather(city)
        print("Cuaca di Jakarta:")
        for k, v in result.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())