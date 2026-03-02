## Property Intelligence Hub

An application based on Retrieval-Augmented Generation (RAG) architecture that lets you extract insights from real estate web articles using natural language queries. Built with tools such as LangChain, Groq LLaMA, ChromaDB, and Streamlit.


🔗 **Live Demo:** [propintelai.streamlit.app](https://propintelai.streamlit.app/)

---

 ## App Preview

<img width="1449" height="503" alt="image" src="https://github.com/user-attachments/assets/f3643692-cceb-4325-b5f1-c2c06eca916d" />


> The sidebar lets you paste up to 3 article URLs and trigger ingestion status in real time. The main panel provides a question input and displays AI-generated answers alongwith sources.

---

## Features

- Scrape and ingest content from up to 3 URLs
- Embeds documents using `sentence-transformers/all-mpnet-base-v2`
- Stores and retrieves chunks via ChromaDB vector store
- Answers questions using Groq's `llama-3.3-70b-versatile` model
- Clean Streamlit UI for non-technical users

--
## Architecture
<img width="852" height="625" alt="image" src="https://github.com/user-attachments/assets/f92499a9-3881-480b-ba09-1164255641e4" />


## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace (`all-mpnet-base-v2`) |
| Vector Store | ChromaDB |
| Orchestration | LangChain (LCEL) |
| Web Scraping | LangChain `WebBaseLoader` |
| Frontend | Streamlit |

--

Setup & Installation
1. Clone the repository
bashgit clone (https://github.com/Gari1428/property_intelligence_hub.git)
cd property-intelligence-hub
2. Create and activate a virtual environment
bashpython -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
3. Install dependencies
bashpip install -r requirements.txt
4. Set up environment variables
Create a .env file in the project root:
envGROQ_API_KEY=your_groq_api_key_here

🔑 Get your free Groq API key at console.groq.com

## Usage

- Enter URLs — Paste up to 3 article URLs in the sidebar (e.g., CNBC, Zillow, Redfin)
- Process URLs — Click the button to scrape, chunk, embed, and store the content
- Ask a Question — Type any natural language question in the input box
- Get an Answer — Receive a concise response with source attribution

--


## Configuration

Can tweak the following constants in `rag.py`:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Size of each text chunk |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | HuggingFace embedding model |
| `COLLECTION_NAME` | `real_estate` | ChromaDB collection name |
| `temperature` | `0.9` | LLM response creativity |
| `max_tokens` | `500` | Maximum tokens in LLM response |

---

 ## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [LangChain](https://www.langchain.com/) for the RAG orchestration framework
- [Groq](https://groq.com/) for ultra-fast LLM inference
- [ChromaDB](https://www.trychroma.com/) for the local vector store
- [Streamlit](https://streamlit.io/) for the rapid UI framework
