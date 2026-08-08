import os
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

def get_text_content(response_obj):
    """Safely extracts text content from LLM response whether it's a string or a list."""
    content = response_obj.content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif hasattr(part, "text"):
                text_parts.append(part.text)
        return "".join(text_parts).strip()
    return str(content).strip()

def test_rag_pipeline():
    print("Loading Database, Web Search, and AI model... Please wait.\n")
    
    # Loading the local database and embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Initializing the Gemini model with a low temperature for factual responses and web search fallback
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    web_search = DuckDuckGoSearchRun(max_results=3)

    print("✅ Smart Agent (with Web Fallback) is Online! (Type 'exit' to quit)\n")
    print("-" * 50)
    
    while True:
        query = input("\nAsk about a scheme (or just say hi!): ")
        
        if query.lower() in ['exit', 'quit']:
            print("Shutting down agent. Great job today!")
            break
            
        if not query.strip():
            continue
            
        print("Thinking...")
        try:
            # Get relevant schemes from the local database
            local_docs = retriever.invoke(query)
            local_context = "\n\n".join([doc.page_content for doc in local_docs])
            
            # Smart Routing Prompt: Decide whether to answer from local DB or search the web
            routing_prompt = f"""
            You are a helpful Government Schemes Assistant. 
            
            User's Query: "{query}"
            
            Local Database Context:
            {local_context}
            
            INSTRUCTIONS:
            1. If the user's query is a simple greeting or small talk (like "hi", "hello", "how are you"), just reply naturally and be friendly.
            2. If the user is asking about a scheme, check the Local Database Context. If the answer is there, provide a beautifully formatted answer.
            3. If the user is asking a factual question or about a scheme that is NOT in the Local Database Context, DO NOT attempt to answer. Instead, output exactly this single word and nothing else: SEARCH_WEB
            """
            
            # Safely extract Gemini's decision
            raw_response = llm.invoke(routing_prompt)
            response = get_text_content(raw_response)
            
            # Web Search Fallback: If Gemini indicates the answer isn't in the local DB, perform a live web search
            if response.strip() == "SEARCH_WEB":
                print("🌐 Scheme not found locally. Searching the live web...")
                
                # Run the DuckDuckGo search
                web_results = web_search.invoke(query)
                
                # Ask Gemini to answer using the brand new web data
                web_prompt = f"""
                You are a helpful Government Schemes Assistant. 
                Answer the user's query using the following web search results. 
                Please start your response by mentioning: *"I couldn't find this in my verified local database, but based on a live web search:"*
                
                User's Query: "{query}"
                Web Results: {web_results}
                """
                raw_web_response = llm.invoke(web_prompt)
                final_answer = get_text_content(raw_web_response)
                
                print("\n🤖 GEMINI SAYS:")
                print(final_answer)
                print("-" * 50)
                
            else:
                # Gemini found it in the local database OR it was just small talk!
                print("\n🤖 GEMINI SAYS:")
                print(response)
                print("-" * 50)
                
        except Exception as e:
            print(f"\n❌ Oops, an error occurred: {e}")

if __name__ == "__main__":
    test_rag_pipeline()