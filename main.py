import os
import logging
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langserve import add_routes
from pydantic import BaseModel
import uvicorn
from typing import Optional, List, Any, Dict
from groq import Groq
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from langchain_core.runnables import RunnableSequence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangChain Server with Groq",
    version="1.0",
    description="API server using Groq and FastAPI",
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Verify Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

class GroqLLM(BaseLLM):
    model_name: str = "llama3-8b-8192" 
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        client = Groq(api_key=GROQ_API_KEY)
        generations = []
        
        for prompt in prompts:
            try:
                logger.info(f"Sending prompt to Groq: {prompt[:100]}...")
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    **kwargs
                )
                
                if completion.choices and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                    logger.info(f"Received response from Groq: {response[:100]}...")
                    generations.append([Generation(text=response)])
                else:
                    error_msg = "Empty response from Groq API"
                    logger.error(error_msg)
                    generations.append([Generation(text=error_msg)])
                    
            except Exception as e:
                error_msg = f"Groq API error: {str(e)}"
                logger.error(error_msg)
                generations.append([Generation(text=error_msg)])
        
        return LLMResult(generations=generations)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

# Initialize LLM
llm = GroqLLM()

# Test Groq connection immediately
def test_groq_connection():
    try:
        test_prompt = "Hello, can you hear me?"
        logger.info("Testing Groq connection...")
        test_result = llm.generate([test_prompt])
        logger.info(f"Connection test response: {test_result.generations[0][0].text[:200]}")
    except Exception as e:
        logger.error(f"Groq connection test failed: {str(e)}")
        raise RuntimeError("Failed to connect to Groq API") from e

test_groq_connection()

# Define chains
prompt_template = PromptTemplate.from_template(
    "Tell me current stock price of staock {topic} , give p/e , p/b , volume , moving average and  assuming yourself as a reccomnder tell whether to hold/ buy/ sell stock at this time ."
)

chain = RunnableSequence(prompt_template, llm)

# Add LangServe routes
add_routes(
    app,
    chain,
    path="/langchain",
)

# Stock Price Model
class StockQuery(BaseModel):
    symbol: str
    info_type: Optional[str] = "price"

# Stock Price Route
@app.post("/stock")
async def get_stock_info(stock_query: StockQuery):
    symbol = stock_query.symbol.upper()
    info_type = stock_query.info_type
    
    try:
        if info_type == "price":
            prompt = f"""What is the current price of {symbol} stock? 
            Respond with JSON format: 
            {{"symbol": "{symbol}", "price": number, "currency": "USD", "source": "simulated"}}
            """
            result = await llm.agenerate([prompt])
            response = result.generations[0][0].text
            
            try:
                import json
                return json.loads(response.strip())
            except json.JSONDecodeError:
                return {
                    "symbol": symbol,
                    "price": "Unknown",
                    "currency": "USD",
                    "source": "simulated",
                    "raw_response": response
                }
        else:
            prompt = f"Give me a brief {info_type} about {symbol} stock."
            result = await llm.agenerate([prompt])
            return {
                "symbol": symbol,
                "info_type": info_type,
                "response": result.generations[0][0].text
            }
    except Exception as e:
        logger.error(f"Error in stock endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Server with Groq"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)