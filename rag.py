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

def process_urls(urls):
    """
    This function scraps a data from a url and srores it in a vector database.
   
    """

   
    print("Loading data")
    loader = WebBaseLoader(
        urls
    )
    data = loader.load()

    
if __name__ == "__main__":
    urls = ["https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]


    process_urls(urls)
    