import validators
import streamlit as st
from langchain_classic.prompts import PromptTemplate  # FIXED: Use standard langchain
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# Streamlit App Config
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

with st.sidebar:
    groq_api_key = st.text_input('Groq API Key', value="", type='password')

generic_url = st.text_input("URL", label_visibility="collapsed")

# 1. Map-Reduce Prompts
map_prompt_template = """
Write a summary of the following content:
"{text}"
Summary:
"""
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

combine_prompt_template = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

if st.button('Summarize the content from YT or website'):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error('Please provide the information to get Started')
    elif not validators.url(generic_url):
        st.error('Please enter a valid Url.')
    else:
        # Step 1: Initialize Model
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        except Exception as e:
            st.error(f"Error initializing Groq: {e}")
            st.stop()

        # Step 2: Load Content
        try:
            with st.spinner('Loading content...'):
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                
                docs = loader.load()
                
                # Check if documents are empty before proceeding
                if not docs:
                    st.error("Error: No content found. If this is a YouTube video, it might not have a transcript/captions available.")
                    st.stop()
                    
        except Exception as e:
            st.error(f"Error loading URL: {e}")
            st.stop()

        # Step 3: Summarize
        try:
            with st.spinner('Summarizing...'):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=200)
                final_docs = text_splitter.split_documents(docs)

                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                    verbose=True
                )
                output_summary = chain.run(final_docs)
                st.success(output_summary)
                
        except Exception as e:
            st.exception(f"Error during summarization: {e}")