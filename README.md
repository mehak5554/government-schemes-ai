## 🏛️ Government Schemes AI

An Agentic RAG-based AI chatbot designed to help users discover and understand Indian government schemes.

The system first searches a curated local database of 35 government schemes using semantic retrieval. If the required information is not available or sufficiently relevant in the local database, the system automatically falls back to live web search using DuckDuckGo and uses Gemini to generate the final response.

## ✨ Features

- 🔎 Semantic search over a curated database of 35 government schemes
- 🧠 RAG (Retrieval-Augmented Generation) using ChromaDB
- 🤖 Gemini LLM for natural-language responses
- 🌐 Live web-search fallback using DuckDuckGo
- 🔗 Displays web sources when information is obtained through web search
- 💬 Interactive Streamlit chatbot interface
- ⚡ Cached AI resources for improved application performance
- 🗃️ Local vector database for scheme retrieval

## 🏗️ System Architecture

                    User Query
                        │
                        ▼
                Streamlit Chat UI
                        │
                        ▼
                 Query Processing
                        │
                        ▼
              Chroma Vector Database
                        │
                  Retrieve Top 3
                   Relevant Chunks
                        │
                        ▼
                 Gemini LLM Router
                    /        \
                   /          \
          Relevant Context    Not Relevant
                 │                │
                 ▼                ▼
          Local DB Answer    DuckDuckGo Search
                                  │
                                  ▼
                              Web Results
                                  │
                                  ▼
                              Gemini LLM
                                  │
                                  ▼
                           Final Answer + Sources 

## 🛠️ Technologies Used 

Technology	Purpose
Python	    Core programming language
Streamlit	Web-based chatbot interface
LangChain	RAG and LLM integration
ChromaDB	Vector database
HuggingFace Embeddings	Text embeddings
Gemini	    Large Language Model
DuckDuckGo	Live web search
Pandas	    Dataset handling
dotenv	    Environment variable management 

## 📂 Project Structure 

government-schemes-ai/
│
├── app.py
├── agent.py
├── create_db.py
├── direct_test.py
├── test.py
├── schemes_data.csv
├── packages.txt
├── chroma_db/
│   └── Vector database files
│
├── .gitignore
└── README.md 

## 🔄 How It Works 

1. Local Knowledge Retrieval

The user's query is converted into an embedding using the all-MiniLM-L6-v2 embedding model.

ChromaDB then retrieves the most relevant scheme documents from the local database.

2. LLM-Based Routing

Gemini evaluates the retrieved context against the user's query.

If the retrieved information is relevant and contains the required answer, Gemini generates the response using the local database.

3. Web Search Fallback

If the required information is not sufficiently available in the local database, the system triggers a DuckDuckGo web search.

The retrieved web results are then provided to Gemini, which generates the final response and displays the available sources.

4. Streamlit Interface

The complete system is exposed through a Streamlit chatbot interface where users can interact with the assistant conversationally.

📊 Dataset

The project uses a curated dataset containing information about 35 Indian government schemes.

The dataset contains scheme-related information used by the retrieval system to provide relevant responses.

🔐 Environment Variables

The Gemini API key is stored locally using a .env file.

Example:
GOOGLE_API_KEY=your_api_key_here 
The .env file is excluded from the Git repository using .gitignore. 

## 🚀 Running the Project 

1. Clone the repository 
git clone https://github.com/mehak5554/government-schemes-ai.git
cd government-schemes-ai 

2. Install dependencies

Install the required Python packages according to the project's dependency file.

3. Configure the API key

Create a .env file in the project directory and add your Gemini API key:
GOOGLE_API_KEY=your_api_key_here 

4. Run the Streamlit application
streamlit run app.py 

The chatbot will then open in your browser.

## 🎯 Project Objective 

The objective of this project is to develop an intelligent assistant that combines Retrieval-Augmented Generation, Large Language Models, vector search, and web-search tools to provide users with accessible information about Indian government schemes. 

## 👩‍💻 Author

Mehak Thakur

B.Tech CSE (AI & ML)