import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title="Summarize Text - HF", page_icon="🦜")
st.title("Summarize text from YouTube or a Website - HF")
st.subheader('Provide the URL')

with st.sidebar:
    hf_token=st.text_input("Enter your HuggingFace token",type="password",value="")

url=st.text_input("Enter URL",label_visibility="collapsed")

if hf_token:
    repo_id="mistralai/Mistral-7B-Instruct-v0.3"
    llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=hf_token)

template="""
Provide a summary of the following content
Content:{text}
"""

prompt=PromptTemplate(template=template,input_variables=["text"])

if st.button("Summarize the content from YouTube or Website URL"):
    if not hf_token.strip() or not url.strip():
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