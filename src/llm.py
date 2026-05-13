import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm(provider: str = "openai", model: str = None, temperature: float = 0.7) -> ChatOpenAI:
    """
    Returns a configured LangChain Chat model.
    By default, uses OpenAI. Can switch to OpenRouter by passing provider="openrouter".
    """
    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "openrouter":
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=model or "openai/gpt-4o-mini",
            temperature=temperature,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Hybrid Deep Research",
            }
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
