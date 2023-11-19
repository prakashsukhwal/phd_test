import streamlit as st
import sqlite3
import openai
import pandas as pd
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import io
import base64
import os, sys, signal
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
import keys
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
im = Image.open("guru1.png")
st.set_page_config(page_title="DiseaseGuru", page_icon=im)
os.environ["OPENAI_API_KEY"] = keys.OPENAI_API_KEY

@st.cache_resource
def create_embeddings():
    # with os.scandir("pdf2") as it:
    folder = os.listdir("pdf")
    if len(folder) == 1:
        print('directory has only pdf')
        loader = DirectoryLoader('pdf',
                            glob="./*.pdf",
                            loader_cls=PyPDFLoader)

        documents = loader.load()
        print(len(documents))

        #splitting the text into
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80, separators=["\\n\\n"])
        texts = text_splitter.split_documents(documents)
        persist_directory = 'pdf'
        # clean the directory before saving the embeddings to make sure no data from earlier is there
        # !rm -rf /content/drive/MyDrive/KG creation/Medical /Last Lap 2023-24/data and other excels/chroma dbs/disease-BGE_db6

        ## Here is the new embeddings being used
        # embedding = model_norm
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=texts,
                                        embedding=embedding,
                                        persist_directory=persist_directory)
        vectordb.persist()
    else:
        pass


# Create a connection to the database
conn = sqlite3.connect('user.db')
c = conn.cursor()

# Create a table to store user details
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY,
             password TEXT NOT NULL,
             gender TEXT NOT NULL,
             age INTEGER NOT NULL,
             health_conditions TEXT NOT NULL,
             profile_picture BLOB);''')


# Initialize session state
if 'username' not in st.session_state:
    st.session_state['username'] = None


# Registration page
def registration():
    st.header('Registration')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.selectbox('Age', range(20, 101))
    health_conditions = st.multiselect('Existing Health Conditions', ['Diabetes', 'Hypertension', 'Asthma', 'cognitive impairment', 'high cholesterol'])
    profile_picture = st.file_uploader('Upload Profile Picture', type=['jpg', 'jpeg', 'png'])
    if st.button('Register'):
        # Check if user already exists
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        if c.fetchone() is not None:
            st.error('User already exists. Please proceed to login.')
        else:
            # Add user details to database
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)", (username, password, gender, age, ', '.join(health_conditions), profile_picture.read()))
            conn.commit()
            st.success('Registration successful. Please proceed to login.')
# Login page
def login():
    st.header('Login')
    username = st.text_input('Username', key='111')
    password = st.text_input('Password', type='password', key='222')
    if st.button('Login'):
        # Check if user exists
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        if c.fetchone() is not None:
            st.success('Login successful.')
            st.session_state['username'] = username
        else:
            st.error('Invalid username or password. Please try again.')
# User details page
def user_details():
    st.header('User Profile')
    c.execute("SELECT * FROM users WHERE username=?", (st.session_state['username'],))
    user = c.fetchone()
    user_data = {
        'Username': [user[0]],
        'Gender': [user[2]],
        'Age': [user[3]],
        'Existing Health Conditions': [user[4]],
        'Profile Picture': [f'<img src="data:image/jpeg;base64,{base64.b64encode(user[5]).decode()}" style="float:right;width:100px;height:100px;">']
    }
    df = pd.DataFrame(user_data)
    # df = df.set_index('Username')
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# Logout functionality
def logout():
    st.session_state['username'] = None
    st.success('Logout successful.')
    st.session_state.steps = {}
    
    

def app():
    # im = Image.open("guru1.png")
    # st.set_page_config(page_title="DiseaseGuru", page_icon=im)
    with open("guru1.png", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-20%;margin-left:20%;">
            <img src="data:image/png;base64,{data}" width="100" height="150">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title(f'''DiseaseGuru: Your Personal Healthcare Agent''')
    # a catchy line for our DiseaseGuru    
    st.sidebar.write('''I'm DiseaseGuru, your virtual health companion, 
                    here to guide you through the maze of diseases with knowledge and care.''')
    menu = ['Home', 'Registration', 'Login', 'User Details']
    choice = st.sidebar.selectbox('Select a page', menu)
    if choice == 'Home':
        st.write('Welcome to the User Registration and Login App!')
    elif choice == 'Registration':
        registration()
    elif choice == 'Login':
        login()
    elif choice == 'User Details':
        if st.session_state['username'] is not None:
            user_details()

            # start the chat        

            os.environ["OPENAI_API_KEY"] = keys.OPENAI_API_KEY
            # loading the vectordb later on for future use
            PERSIST_DIRECTORY = 'pdf'#'disease-openaidb12'

            @st.cache_resource
            def get_embeddings():
                embeddings = OpenAIEmbeddings() # here we can use open source as well
                return embeddings

            @st.cache_resource
            def get_vdb():    
                # embeddin = model_norm
                embeddin = get_embeddings()#OpenAIEmbeddings()
                vectordb2 = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddin)
                return vectordb2

            @st.cache_resource(ttl="1h")
            def configure_retriever():
                # Read the vector DB  
                vectordb = get_vdb() 
                # Define retriever
                retriever = vectordb.as_retriever(search_kwargs={"k": 4})
                return retriever


            tool_1 = create_retriever_tool( configure_retriever(),
                "search_disease_info",
                "Searches and returns documents regarding the disease information."
                )
            tools = [tool_1]
            # another tool
            # Define which tools the agent can use to answer user queries
            search = DuckDuckGoSearchRun()
            def duck_wrapper(input_text):
                try:
                    search_results1 = search.run(f"webmd.com {input_text}")
                    search_results2 = search.run(f"mayoclinic.org {input_text}")
                    search_results = ".\n".join((search_results1, search_results2))[:500]
                except:
                    search_results = "no results"
                return search_results

            tools.append(
                    Tool(
                    name = "search_websites",
                    func=duck_wrapper,
                    description="Must use this if you cannot answer disease related questions with tool_1 search_disease_info"
                )
            )
            # anther tool to keep the cahtbot in certain domains
            def other_ques(input_text):
                return("Sorry this question is out of scope for me. please ask something related to diseases.")
            tools.append(
                    Tool(
                    name = "out_of_scope",
                    func=other_ques,
                    description = "Use this when user asks a question on anything other than diseases or user details"
                    # description="use this when you cannot answer a question using tools: search_disease_info or search_web."
                )
            )
            llm = ChatOpenAI(temperature=0.0,
                            #  model_name='ft:gpt-3.5-turbo-0613:personal:diseaseguru-test:7vPRlxp2',
                                model_name="gpt-3.5-turbo-16k", streaming=True
                            )
            
            # This is needed for both the memory and the prompt
            memory_key = "history"
            msgs = StreamlitChatMessageHistory(key=memory_key)
            
            memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, chat_memory=msgs)
           

            # @st.cache_resource
            def sys_prompt():
                
                c.execute("SELECT * FROM users WHERE username=?", (st.session_state['username'],))
                user = c.fetchone()
                user_name = user[0]
                # user_profile = f'{user_name}, {user_age} yrs old {user_gender} with { " ".join(str(i) for i in preconditions)} '
                user_profile = f'{user[0]}, {user[3]} yrs old {user[2]} with {user[4]} '
                # print(user_profile)

                system_message = SystemMessage(
                        content=( f'''You are DiseaseGuru, an AI agent, talking to me (the user), {user_profile} "
                                 "never use more than 120 words in answers"
                                 "Only answer questions about diseases and human user."
                        "You only answer disease related questions using {user_profile}. Politely say 'sorry' to any other questions."
                            "Always use the given tools first to answer user questions"
                            
                            "Always give short answers only in less than 120 words."
                            "Do not provide extra or additional information in your answer."
                            "Do not provie any false information"
                            "Feel free to use given tools to look up relevant information. Then use only the relevant information to answer the question. "
            "Here are some sample conversations between the Assistant and a User:

            User: Who am i?
            Assistant: you are Amit

            User: Hey?
            Assistant: Hello Amit, What questions do you have today?
                     
            User: Who is prone to menstrual cramps?
            Assistant: females, but you are a male so you ar not prone to it.

            User: what is salman khan?
            Assistant: I am sorry Amit but I can only answer disease related questions.
            
            User: what is ajpj?
            Assistant: I am sorry Amit but I don't understand your question. if your question is regarding a disease, can you rephrase it?

            User: what is NATO?
            Assistant: I am sorry Amit but I can only answer disease related questions.

            User: can i befirend someone?
            Assistant: I am sorry Amit but I can only answer disease related questions.

            User: what is diabetes? provide a short answer in less than 100 words.
            Assistant: Diabetes is a condition where the blood sugar (glucose) levels in the body are too high. It occurs when the pancreas does not produce enough insulin or when the body does not respond properly to insulin. "
            User: Bbye
            Assistant: Good Bye dear Amit, take care! If you have anymor equestions then feel free to ask.
                        '''
                        )
                )
                return system_message

            prompt = OpenAIFunctionsAgent.create_prompt(
                    system_message=sys_prompt(),
                    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)])


            @st.cache_resource
            def llm_agent():
                agent = OpenAIFunctionsAgent(llm=llm,
                                        tools=tools,
                                        prompt=prompt
                                        )
                return agent

            agent_executor = AgentExecutor(agent=llm_agent(), 
                                        tools=tools, 
                                        memory=memory, 
                                        verbose=True,
                                        return_intermediate_steps=True)
            c.execute("SELECT * FROM users WHERE username=?", (st.session_state['username'],))
            user = c.fetchone()
            # Re-setting session state 

            if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
                msgs.clear()
                msgs.add_ai_message(f"How can I help you, {user[0]}")
                if not st.session_state:
                    pass
                else:
                    st.session_state.steps = {}
                # logout()
            # Logout functionality
            

            view_messages = st.sidebar.expander("View the chat history here anytime!")
            # Render current messages from StreamlitChatMessageHistory
            for msg in msgs.messages:
                st.chat_message(msg.type).write(msg.content)

            # If user inputs a new prompt, generate and draw a new response
            if prompt:= st.chat_input(placeholder = "Share your disease related questions"):
                st.chat_message("human").write(prompt)
                # Note: new messages are saved to history automatically by Langchain during run
                response = agent_executor({"input": prompt})["output"]#llm_chain.run(prompt)
                st.chat_message("ai").write(response)

            # Draw the messages at the end, so newly generated ones show up immediately
            with view_messages:
                view_messages.json(st.session_state.history)
            if st.sidebar.button("logout"):
                st.session_state['username'] = None
                st.success('Logout successful.')
                
                msgs.clear()
                st.session_state.clear()
        else:
            st.warning('Please login to view user details.')


if __name__ == '__main__':
    create_embeddings()
    app()
