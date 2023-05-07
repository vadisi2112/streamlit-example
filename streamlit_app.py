from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

from pprint import pprint
import json
from langchain import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
from io import StringIO
import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import nltk
import os
import openai
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the most recent state of the union address.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions related to Crypto News.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
In your answer, recommend places to find more information.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to Crypto News, politely inform them that you are tuned to only answer questions about the news setup.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[ "question", "context"])
        
os.environ['OPENAI_API_KEY']= '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'
​openai.api_key = os.environ["OPENAI_API_KEY"]

gpt4 = ChatOpenAI(model="gpt-4", temperature=0, request_timeout=300)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

openai.api_key = os.environ["OPENAI_API_KEY"]

def ingestion(id):
    # Load Data
    loader = UnstructuredFileLoader(f"{id}_output.txt")
    raw_documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)


    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)


    # Save vectorstore
    with open(f"vectorstore{id}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
        
def get_chain(vectorstore):
    # print(QA_PROMPT)
    # print(CONDENSE_QUESTION_PROMPT)
    model = 'gpt-4'
    llm = ChatOpenAI(model=model, temperature=0, request_timeout=300)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        # qa_prompt=QA_PROMPT,
        # condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain


import pickle

def ask_question(id='1', question=None, chat_history=None):
    with open(f"vectorstore{id}.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore)
    result = qa_chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return result["answer"].strip(), chat_history

def answer_question(docs, question, method="refine", llm=gpt4):
    chain = load_qa_chain(llm, chain_type=method, verbose=True)
    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    return response["output_text"]
​
​
def summary(input_text, use_gpt4=True):
    model = 'gpt-4' if use_gpt4 else 'gpt-3.5-turbo'
    chunk_size = 16000 if use_gpt4 else 8000
    llm = ChatOpenAI(model=model, temperature=0, request_timeout=300)

    docs = RecursiveCharacterTextSplitter(chunk_size=chunk_size).create_documents(texts=[input_text])

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    return chain.run(docs)
​
​
def main():
    st.title("Document Summarizer and Question Answering")
​
    operation = st.radio(
        "Choose an operation:", [ "Read and answer question", "Read and summarise"]
    )
​
    if operation == "Read and answer question":
        question = st.text_input("Question:")
​
    model = st.selectbox("model:", ["gpt-4", "gpt-3.5-turbo"], index=0)
​
    if st.button("Submit"):
        with st.spinner("Processing..."):
            chat_history = []

            answer, chat_history = ask_question(question=question, chat_history=chat_history)
            if operation == "Read and summarise":
                f = open(f"1_output.txt", "r")
                input_text = " ".join(f.readlines()).replace("\n\n", " ")
                result = summary(input_text)
            else:
                result = ask_question(question=question, chat_history=chat_history)
            st.subheader("Output:")
            st.write(result)
​

