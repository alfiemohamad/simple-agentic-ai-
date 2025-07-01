# AI Assistant with Weather Tools

AI Assistant berbasis LangChain yang dapat memberikan informasi cuaca dengan memory persisten menggunakan Neo4j. Assistant ini menggunakan ReAct (Reasoning and Acting) pattern untuk menjalankan tools dan menyimpan riwayat percakapan.

## 📋 Fitur

- ✅ **Weather Information**: Mendapatkan cuaca saat ini dan prediksi 5 hari
- ✅ **Persistent Memory**: Menyimpan riwayat percakapan di Neo4j
- ✅ **ReAct Agent**: Menggunakan pola ReAct untuk reasoning dan action
- ✅ **Async Support**: Full async/await support
- ✅ **REST API**: FastAPI untuk interface
- ✅ **Error Handling**: Robust error handling dan logging

## 🏗️ Struktur Proyek

```
.
├── app/
│   ├── __main__.py          # Entry point aplikasi
│   ├── agent.py             # ReAct agent dan wrapper
│   ├── config.py            # Konfigurasi environment
│   ├── memory.py            # Neo4j memory handler
│   ├── server.py            # FastAPI server
│   ├── tools/
│   │   ├── __init__.py
│   │   └── weather.py       # Weather API tools
│   └── utils/
│       ├── __init__.py
│       └── error.py         # Custom exceptions
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md               # Dokumentasi ini
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database
- OpenWeatherMap API Key
- Ollama dengan model llama3

### 1. Clone & Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd simple-agentic-ai

# Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Neo4j Database

#### Option A: Docker (Recommended)
```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v neo4j_data:/data \
    -v neo4j_logs:/logs \
    -v neo4j_import:/var/lib/neo4j/import \
    -v neo4j_plugins:/plugins \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

#### Option B: Neo4j Desktop
1. Download dari [Neo4j Desktop](https://neo4j.com/download/)
2. Install dan buat database baru
3. Set password untuk user `neo4j`

### 3. Setup Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download model llama3
ollama pull llama3

# Verify model
ollama list
```

### 4. Get OpenWeatherMap API Key

1. Daftar di [OpenWeatherMap](https://openweathermap.org/api)
2. Dapatkan API key gratis
3. Simpan API key untuk step selanjutnya

### 5. Configure Environment Variables

Buat file `.env` di root directory:

```bash
# OpenWeatherMap API Key
OWM_API_KEY=your_openweathermap_api_key_here

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Agent Configuration
AGENT_TEMPERATURE=0.7
```

### 6. Run Application

```bash
# Jalankan aplikasi
python -m app

# Atau dengan uvicorn langsung
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

Server akan berjalan di `http://localhost:8000`

## 🧪 Testing API

### Health Check
```bash
curl http://localhost:8000/health
```

### Send Chat Message
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"text": "cuaca jakarta hari ini"}'
```

### Get Conversation History
```bash
curl http://localhost:8000/memory
```

### Example Responses

**Chat Response:**
```json
{
  "status": "success",
  "input": "cuaca jakarta hari ini",
  "output": "Weather in Jakarta, ID:\n- Temperature: 31.62°C\n- Condition: overcast clouds\n- Humidity: 68%\n- Wind: 3.15 m/s",
  "intermediate_steps": [...]
}
```

**Memory Response:**
```json
{
  "status": "success",
  "history": [
    "[2025-07-01T10:30:00Z]\nUser: cuaca jakarta hari ini\nAgent: Weather in Jakarta, ID:..."
  ],
  "count": 1
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OWM_API_KEY` | OpenWeatherMap API Key | - | ✅ |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` | ✅ |
| `NEO4J_USER` | Neo4j username | `neo4j` | ✅ |
| `NEO4J_PASSWORD` | Neo4j password | `password` | ✅ |
| `AGENT_TEMPERATURE` | LLM temperature | `0.7` | ❌ |

### Customizing Memory

Untuk mengubah jumlah riwayat yang disimpan:

```python
# Di app/agent.py
long_term_mem = Neo4jMemory(
    settings.NEO4J_URI,
    settings.NEO4J_USER,
    settings.NEO4J_PASSWORD,
    k=10  # Ubah dari 5 ke 10
)
```

### Adding New Tools

1. Buat tool baru di `app/tools/`:

```python
# app/tools/my_tool.py
from langchain_core.tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Description of what this tool does"
    
    async def _arun(self, input_param: str) -> str:
        # Tool logic here
        return "Tool result"
    
    def _run(self, input_param: str) -> str:
        raise NotImplementedError("Sync execution not supported")
```

2. Import dan tambahkan ke tools list di `app/agent.py`:

```python
from app.tools.my_tool import MyCustomTool

tools = [CurrentWeatherTool(), ForecastTool(), MyCustomTool()]
```

## 🏗️ Architecture

### Components

1. **FastAPI Server** (`server.py`): REST API endpoints
2. **Agent Wrapper** (`agent.py`): LangChain ReAct agent wrapper
3. **Memory Handler** (`memory.py`): Neo4j integration untuk persistent memory
4. **Weather Tools** (`tools/weather.py`): OpenWeatherMap API integration
5. **Configuration** (`config.py`): Environment variables management

### Data Flow

```
User Request → FastAPI → AgentWrapper → ReAct Agent → Tools → Response
                                     ↓
                               Neo4j Memory
```

### Memory Storage

Setiap percakapan disimpan di Neo4j dengan struktur:

```cypher
(:Memory {
  user_input: "cuaca jakarta hari ini",
  agent_output: "Weather in Jakarta...",
  timestamp: datetime(),
  created_at: "2025-07-01T10:30:00"
})
```

## 🛠️ Development

### Running in Development Mode

```bash
# Dengan auto-reload
uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

# Dengan debug logging
PYTHONPATH=. python -m app
```

### Testing

```bash
# Test weather API
python -c "
import asyncio
from app.tools.weather import get_current_weather
print(asyncio.run(get_current_weather('Jakarta')))
"

# Test Neo4j connection
python -c "
from app.memory import Neo4jMemory
from app.config import settings
mem = Neo4jMemory(settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD)
mem.add('test', 'response')
print(mem.load())
"
```

### Debugging Tips

1. **Check Ollama status**: `ollama list`
2. **Check Neo4j logs**: `docker logs neo4j`
3. **Verify API key**: Test OpenWeatherMap API directly
4. **Monitor logs**: Watch console output untuk error messages

## 🚨 Troubleshooting

### Common Issues

**1. "Neo4j connection failed"**
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Check connection
telnet localhost 7687
```

**2. "OWM API Key missing"**
- Verify `.env` file exists dan berisi `OWM_API_KEY`
- Check API key valid di OpenWeatherMap dashboard

**3. "Ollama model not found"**
```bash
# Pull model again
ollama pull llama3

# Check available models
ollama list
```

**4. "Agent stuck in loop"**
- Check Ollama model response format
- Verify tool descriptions are clear
- Check agent prompt template

**5. "Memory not saving"**
- Check Neo4j credentials
- Verify database is writable
- Check logs for Neo4j errors

### Performance Optimization

1. **Adjust Neo4j memory**: Increase heap size untuk large datasets
2. **Limit conversation history**: Reduce `k` parameter di memory
3. **Optimize Ollama**: Use smaller model untuk faster response
4. **Cache API responses**: Implement caching untuk frequent requests

## 📚 API Reference

### Endpoints

#### POST /chat
Send message to AI assistant

**Request:**
```json
{
  "text": "cuaca jakarta hari ini"
}
```

**Response:**
```json
{
  "status": "success",
  "input": "cuaca jakarta hari ini",
  "output": "Weather information...",
  "intermediate_steps": []
}
```

#### GET /memory
Get conversation history

**Response:**
```json
{
  "status": "success",
  "history": ["conversation1", "conversation2"],
  "count": 2
}
```

#### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "message": "AI Assistant API is running"
}
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Submit Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

Jika mengalami masalah:

1. Check troubleshooting section
2. Review logs untuk error messages
3. Verify semua prerequisites terinstall
4. Create issue di repository dengan detail:
   - Error message
   - Steps to reproduce
   - Environment information
   - Log output

## 🔮 Roadmap

- [ ] Add more weather data sources
- [ ] Implement user authentication
- [ ] Add conversation threading
- [ ] Web UI interface
- [ ] Docker compose setup
- [ ] Kubernetes deployment
- [ ] Monitoring dan metrics
- [ ] Rate limiting
- [ ] API key management
- [ ] Multi-language support