import sys
import streamlit as st
from src.model import qa_chain


st.set_page_config(page_title="Potentiam Chatbot", page_icon=":robot_face:")


st.title('Welcome To Potentiam')


text = st.text_input("Please enter your Query")
if st.button('Submit'):
    if not text:
        st.warning("Please enter your Query")
    else:
        with st.spinner("Please wait, it may take some time to run..."):
            result=qa_chain.invoke(text)
            response=result['result']
            st.write(response)