import streamlit as st
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="Gov Schemes AI",
    page_icon="🏛️",
    layout="centered"
)

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

@st.cache_resource
def load_ai_agent():
    # Loading the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Loading the Chroma vector store from the persisted directory
    vector_store = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings
    )
    
    # Retrieves the top 3 documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Gemini initialization with a low temperature for factual responses
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
    
   
    web_search = DuckDuckGoSearchResults()

    return retriever, llm, web_search

# Load cached resources
retriever, llm, web_search = load_ai_agent()

# sidebar UI
with st.sidebar:
    st.title("🏛️ Gov Schemes AI")
    st.markdown("This Agentic RAG system searches a **curated local database of 35 government schemes** first.")
    st.markdown("If the answer isn't found locally, it automatically falls back to live web search using DuckDuckGo.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# Main UI
st.title("Government Schemes Assistant")
st.caption("Ask me about any government scheme!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about a scheme..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show assistant thinking
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get relevant schemes from the local database
                local_docs = retriever.invoke(prompt)
                local_context = "\n\n".join([doc.page_content for doc in local_docs])
                routing_prompt = f"""
                You are a helpful Government Schemes Assistant. 
                
                User's Query: "{prompt}"
                
                Local Database Context:
                {local_context}
                
                INSTRUCTIONS:
                1. If the user's query is a simple greeting or small talk (like "hi", "hello"), reply naturally.
                2. If the user is asking about a scheme, strictly evaluate the Local Database Context. If the context is highly relevant and contains the exact answer, provide a beautifully formatted answer.
                3. If the retrieved context is only vaguely related, or if the actual answer is NOT in the context, DO NOT guess or hallucinate. Instead, output exactly this single word and nothing else: SEARCH_WEB
                """
                
                # Safely extract Gemini's decision
                raw_response = llm.invoke(routing_prompt)
                decision = get_text_content(raw_response)
                
                if decision.strip() == "SEARCH_WEB":
                    
                    # Run the DuckDuckGo search 
                    web_results = web_search.invoke(prompt)
                    
                    # instructions to show Sources
                    web_prompt = f"""
                    You are a helpful Government Schemes Assistant. 
                    Answer the user's query using the following web search results. 
                    
                    Please start your response by mentioning: *"I couldn't find this in my curated local database, but based on a live web search:"*
                    
                    Crucially: At the very end of your response, add a "### Sources" section and list the exact URLs/Links provided in the Web Results so the user can verify the information.
                    
                    User's Query: "{prompt}"
                    Web Results: {web_results}
                    """
                    raw_web_response = llm.invoke(web_prompt)
                    final_answer = get_text_content(raw_web_response)
                    
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    
                else:
                    # Gemini found it in the local database OR it was just small talk
                    st.markdown(decision)
                    st.session_state.messages.append({"role": "assistant", "content": decision})
                    
            except Exception as e:
                error_msg = f"❌ Oops, an error occurred: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})