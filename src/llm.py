import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

load_dotenv()

def get_llm(provider: str = "openrouter", model: str = None, temperature: float = 0.7) -> ChatOpenAI:
    """
    Returns a configured LangChain Chat model.
    By default, uses OpenAI. Can switch to OpenRouter by passing provider="openrouter".
    """
    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-5-nano",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            model_name=model or "nvidia/nemotron-3-super-120b-a12b:free",
            temperature=temperature,
            app_url="http://localhost:8000",
            app_title="Hybrid Deep Research"
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
