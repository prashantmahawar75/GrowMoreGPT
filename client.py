import requests
import json

# Base URL of your FastAPI server
BASE_URL = "http://localhost:8000"

def ask_langchain(topic: str):
    """Query the LangChain endpoint"""
    response = requests.post(
        f"{BASE_URL}/langchain/invoke",
        json={"input": {"topic": topic}}
    )
    return response.json()

def get_stock_price(symbol: str):
    """Query the stock price endpoint"""
    response = requests.post(
        f"{BASE_URL}/stock",
        json={"symbol": symbol, "info_type": "price"}
    )
    return response.json()

if __name__ == "__main__":
    # Example usage
    print("LangChain Example:")
    print(ask_langchain("quantum computing"))
    
    print("\nStock Price Example:")
    print(get_stock_price("AAPL"))