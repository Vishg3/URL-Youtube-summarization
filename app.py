import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title="Summarize Text", page_icon="🦜")
st.title("Summarize text from YouTube or a Website")
st.subheader('Provide the URL')

with st.sidebar:
    groq_api_key=st.text_input("Enter your Groq api key",type="password",value="")

url=st.text_input("Enter URL",label_visibility="collapsed")

if groq_api_key:
    llm=ChatGroq(model="Gemma-7b-It",groq_api_key=groq_api_key)

template="""
Provide a summary of the following content
Content:{text}
"""

prompt=PromptTemplate(template=template,input_variables=["text"])

if st.button("Summarize the content from YouTube or Website URL"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Provide the information to continue")
    elif not validators.url(url):
        st.error("Enter a valid URL")
    else:
        try:
            with st.spinner("Summarizing..."):
                if "youtube.com" in url:
                    loader=YoutubeLoader.from_youtube_url(url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[url],ssl_verify=True,headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
                docs=loader.load()
                chain=load_summarize_chain(llm=llm,chain_type="stuff",prompt=prompt)
                summary=chain.run(docs)
                st.success(summary)
        except Exception as e:
            st.exception(f"Exception: {e}")