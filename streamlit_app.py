import streamlit as st
import requests

st.title("News Summarization fine-tuning")

# 사용자 입력을 받음
input_text = st.text_area("Enter the text you want to summarize:")
max_length = st.slider("Max summary length:", 50, 300, 128)

if st.button('Summarize'):
    if input_text:
        response = requests.post("http://localhost:8000/summarize/",
                                 json = {
                                     "text": input_text,
                                     "max_length": max_length
                                 })
        if response.status_code == 200:
            summary = response.json().get("summary","")
            st.write("### Summary")
            st.write(summary)
        else:
            st.write("Error: ",response.status_code)        
