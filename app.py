import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

# تحميل الـ API Key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# تعريف LLM (Gemini Flash 1.5)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key,
    temperature=0.7
)

# Streamlit framework
st.title('Find Your Favorite Celebrity (Gemini Flash 1.5)')
input_text = st.text_input("Search the topic you want")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"
)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="When was {person} born?"
)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="Mention 5 major events happened around {dob} in the world."
)

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# Chains
chain1 = LLMChain(
    llm=llm, prompt=first_input_prompt,
    verbose=True, output_key='person', memory=person_memory
)

chain2 = LLMChain(
    llm=llm, prompt=second_input_prompt,
    verbose=True, output_key='dob', memory=dob_memory
)

chain3 = LLMChain(
    llm=llm, prompt=third_input_prompt,
    verbose=True, output_key='description', memory=descr_memory
)

parent_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

# Run the chatbot
if input_text:
    output = parent_chain({'name': input_text})
    st.write(output)

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
