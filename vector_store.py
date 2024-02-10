import os
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import constants
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.vectorstores import FAISS
import pickle
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeOG, PodSpec
import pinecone


def store_embeddings(index_name, file_name:str = "OPERATING SYSTEMS.pdf"):
    print("Creating vectorstore...")
    # load document
    loader =  UnstructuredFileLoader(file_name)

    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # # create vectors for similarity search
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    document_search = FAISS.from_documents(texts, embeddings)
    # document_search = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    with open(f"{index_name}.pkl", 'wb') as f:
        pickle.dump(document_search, f)
    return document_search


def load_embedding(index:str = "os-vector", filename:str="OPERATING SYSTEMS.pdf"):
    if not os.path.isfile(f"{index}.pkl"):
        vector = store_embeddings(index_name=index, file_name=filename)
    else:
        with open(f"{index}.pkl", 'rb') as f:
            vector = pickle.load(f)

    return vector


# os.environ['PINECONE_API_KEY'] = constants.PINECONE_API_KEY

# def store_embeddings(index_name, file_name:str = "OPERATING SYSTEMS.pdf"):
#     print("Creating vectorstore...")
#     # load document
#     loader =  UnstructuredFileLoader(file_name)

#     documents = loader.load()

#     # Split text
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)

#     # # create vectors for similarity search
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
   
#     document_search = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    
#     return document_search


# def load_embedding(index:str = "os-vector", filename:str="OPERATING SYSTEMS.pdf"):
#     pc = PineconeOG(api_key=constants.PINECONE_API_KEY)
#     if index not in pc.list_indexes().names():
#         pc.create_index(
#             name=index,
#             metric='cosine',
#             dimension=768,  # Length of embedding
#             spec=PodSpec(
#                 environment="gcp-starter"
#                 )
#         )
#         vector = store_embeddings(index_name=index,file_name=filename)
#     else:
#         print(f"loading Index from pinecone")
#         embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
#         vector = Pinecone.from_existing_index(index, embeddings)

#     return vector
