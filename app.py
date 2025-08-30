## Conversational Q&A Chatbot with Gemini
import streamlit as st

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

## Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

from dotenv import load_dotenv
load_dotenv()
import os


chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    convert_system_message_to_human=True  # علشان يقبل SystemMessage
)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a comedian AI assistant")
    ]

## Function to load Gemini model and get responses
def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

input = st.text_input("Input: ", key="input")
response = get_chatmodel_response(input)

submit = st.button("Ask the question")

## If ask button is clicked
if submit:
    st.subheader("The Response is")
    st.write(response)
