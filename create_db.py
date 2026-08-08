import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def build_vector_db():
    # Reading the CSV file containing government schemes data
    df = pd.read_csv("schemes_data.csv")
    
    # Fill any empty cells to avoid string errors
    df = df.fillna("Not specified")

    documents = []

    # Converting each row of the DataFrame into a Document object for Chroma
    for _, row in df.iterrows():
        content = f"""
Scheme Name: {row['Scheme_name']}
Ministry/Department: {row['Ministry']}
Description: {row['Description']}
Eligibility Criteria: {row['Eligibility']}
Benefits: {row['Benefits']}
Documents Required: {row['Document_Required']}
Official Link: {row['Official_link']}
"""
        doc = Document(
            page_content=content.strip(),
            metadata={"scheme_name": str(row['Scheme_name'])}
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} schemes from CSV.")

    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Storing the documents in a Chroma vector store for efficient retrieval
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("✅ Successfully built and saved Chroma Vector Database in './chroma_db'!")

if __name__ == "__main__":
    build_vector_db()