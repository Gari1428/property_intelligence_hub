from uuid import uuid4
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv


load_dotenv()
CHUNK_SIZE =1000
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vector_store"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)  

    if vector_store is None:
        ef =  HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"trust_remote_code": True})

        vector_store = Chroma(
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),
            collection_name=COLLECTION_NAME
        )


def process_urls(urls):
    """
    This function scraps a data from a url and srores it in a vector database.
   
    """
    print("Initializing components")
    initialize_components()
    vector_store.reset_collection()
    
    print("Loading data")
    loader = WebBaseLoader(
        urls
    )
    data = loader.load()

    print("Split text")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=20
    )
    docs = text_splitter.split_documents(data)
    
    print("Adding documents to vector db")

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)


def generate_answer(query):
    """
    Generate answer using pure LCEL - no langchain.chains dependency
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")
    
    # Get retriever
    retriever = vector_store.as_retriever()
    
    # Create prompt template
    template = """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer the question concisely and accurately."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Get answer
    answer = rag_chain.invoke(query)
    
    # Get source documents separately
    source_docs = retriever.invoke(query)
    sources = ", ".join(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))
    
    return answer, sources
    

if __name__ == "__main__":
    urls = ["https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]


    process_urls(urls)
    answer, sources = generate_answer("Tell me what was the 30 year fixed mortgage rate along with the date?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")

    