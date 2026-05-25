import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from src.llm import get_llm

@tool
async def scrape_webpage(url: str) -> str:
    """
    Scrape text content from a webpage URL.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator="\n")
                # Compress spaces
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)
                return text[:50000] # Limit raw text
            return f"Error: Status {response.status_code}"
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

async def compress_document(text: str, query: str) -> str:
    """
    Compress a long document by extracting only the information relevant to the query.
    """
    if len(text) < 1000:
        return text # Too short to compress
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting information. Extract all facts, figures, and insights from the text that are relevant to the query. Be concise but retain important details."),
        ("user", "Query: {query}\n\nText:\n{text}")
    ])
    
    llm = get_llm(provider="openai", model="gpt-4o-mini", temperature=0.0)
    chain = prompt | llm
    
    # We should ensure text is within context limits (e.g. 20000 chars is usually safe for gpt-4o-mini)
    try:
        result = await chain.ainvoke({"query": query, "text": text})
        return result.content
    except Exception as e:
        return f"Compression error: {str(e)}"
