from uuid import uuid4
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    


if __name__ == "__main__":
    urls = ["https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]


    process_urls(urls)
    