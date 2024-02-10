import langchain
import os
import openai
from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import VectorDBQA, RetrievalQA, ConversationalRetrievalChain
from custom_retrival import CustomRetrivalQA, CustomVectorDBQA, CustomConversationalRetrievalChain
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import constants
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.vectorstores import FAISS
from vector_store import store_embeddings, load_embedding
from typing import Tuple
import streamlit as st


os.environ["HUGGINGFACEHUB_API_TOKEN"] = constants.HF_TOKEN


def get_consersational_chain():
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 64,"max_new_tokens":250})

    file_path = "os-vector"

    document_search = load_embedding(file_path)

    # create a retriver
    retriever = document_search.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold": .75})

    chain = CustomConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, chain_type="map_reduce", return_source_documents=True, verbose=True)
    # chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, chain_type="refine", return_source_documents=True)

    return chain



chat_history = []

def retrieval_aug_gen(chain, question: str, chat_history: list) -> Tuple[str, list]:
    try:
        results = chain.invoke({"question":question, "chat_history": chat_history})
        chat_history.append((question, results['answer']))
    except:
        results = "I don't know the answer"
        chat_history.append((question, "I don't know the answer"))
    
    return results, chat_history


chain = get_consersational_chain()

assitant = st.chat_message("assitant")
assitant.write("Hello How can I help You? You can ask me anything related to OS")

user = st.chat_input("user")
chat_history = []

# Session State also supports the attribute based syntax
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user:
    with st.chat_message("user"):
        st.write(user)
        st.session_state.messages.append({"role": "user", "content": user})
    with st.chat_message("assistant"):
        result,chat_history  = retrieval_aug_gen(chain, user, chat_history)
        if result != "I don't know the answer":
            # src = []
            print(result)
            st.write(f"Answer:- {result['answer']} \n\n")
            st.session_state.messages.append({"role": "assistant", "content": result['answer']})
            try:
                for index, sources in enumerate(result['source_documents']):
                    
                    # src.append(sources.page_content)
                    st.write(f"\n\nSource {index+1}  :- {sources.page_content}")
                    st.session_state.messages.append({"role": "assistant", "content": f"\n\nSource {index+1}  :- {sources.page_content}"})
            except:
                pass
        else:
            st.write(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
