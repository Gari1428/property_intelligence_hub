import streamlit as st
from rag import process_urls, generate_answer

st.title("Property Intelligence Hub")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url.strip() for url in [url1, url2, url3] if url.strip()]
    if not urls:
        placeholder.text("Please enter at least one valid URL to process.")
    else:
       for status in process_urls(urls):
           placeholder.text(status)

query = placeholder.text_input("Ask a Question:")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.header("Sources:")
            for source in sources.split("\n"):
                st.write(f"- {source}")
    except RuntimeError as e:
        placeholder.text("Please process the URLs first before asking a question.")