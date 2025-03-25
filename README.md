# Stock AI Agent

A **FastAPI-based AI agent** that provides stock market analysis using **Groq's ultra-fast LLM inference** and **LangChain** for prompt engineering.

## Features
- **Real-time stock price simulation**
- **Financial metric analysis** (P/E, P/B, volume, moving averages)
- **Buy/Hold/Sell recommendations**
- **RESTful API endpoints**
- **Interactive playground interface**
- **Health monitoring endpoints**

## Prerequisites
- **Python 3.9+**
- **Groq API key** (free at [console.groq.com](https://console.groq.com))
- **pip package manager**

## Installation

### Clone the repository:
```bash
git clone https://github.com/yourusername/stock-ai-agent.git
cd stock-ai-agent
```

### Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Create a `.env` file with your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

## Usage

### Running the Server
```bash
uvicorn main:app --reload
```
The server will start at [http://localhost:8000](http://localhost:8000) with these endpoints:
- **Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **LangChain Playground**: [http://localhost:8000/langchain/playground/](http://localhost:8000/langchain/playground/)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Example API Calls

#### Get Stock Analysis
```bash
curl -X POST "http://localhost:8000/langchain/invoke" \
-H "Content-Type: application/json" \
-d '{"input":{"topic":"AAPL"}}'
```

#### Get Simulated Stock Price
```bash
curl -X POST "http://localhost:8000/stock" \
-H "Content-Type: application/json" \
-d '{"symbol":"MSFT","info_type":"price"}'
```

### Using the Python Client
```python
from client import ask_langchain, get_stock_price

# Get stock analysis
print(ask_langchain("TSLA"))

# Get stock price
print(get_stock_price("GOOGL"))
```

## Deployment

### Docker
```bash
docker build -t stock-ai .
docker run -p 8000:8000 --env-file .env stock-ai
```

### Cloud Platforms
- **AWS**: Deploy to Lambda with API Gateway
- **GCP**: Use Cloud Run for containerized deployment
- **Render/Heroku**: One-click deployment from GitHub

## Configuration
Modify these in `main.py`:
- **`model_name`**: Change Groq model (`"llama3-8b-8192"`, `"mixtral-8x7b"`, etc.)
- **`temperature`**: Adjust creativity/randomness (`0.0-1.0`)
- **Prompt templates** in `PromptTemplate.from_template()`

## Limitations
- Uses **simulated** rather than real market data
- Rate limited by **Groq's free tier** (~30 requests/minute)
- **Llama3-8B** may occasionally hallucinate numbers

## Roadmap
- **Integrate real market data APIs**
- **Add user authentication**
- **Implement response caching**
- **Expand to cryptocurrency analysis**

## Support
For issues, please open an issue on **GitHub**.

## License
MIT License

