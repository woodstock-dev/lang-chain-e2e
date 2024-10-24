# Copyright 2024 Google, LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import chromadb
from chromadb import EmbeddingFunction, Documents

from chromadb.api.models.Collection import Collection
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.embeddings import Embeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores.base import Collection
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils import print_with_time, SuppressStdout
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

DB_DIRECTORY = 'db'
DOCUMENT_SOURCE_DIRECTORY = 'third_party/books'

CHUNK_SIZE=2000
CHUNK_OVERLAP=50
HIDE_SOURCE_DOCUMENTS=False

embeddingModel = OllamaEmbeddings(model="llama3.2")

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return embeddingModel.embed_documents(input)


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

def read_files(native_db):
    """This method loads the PDF files from the source directory
    and uses the explicit PyPDFLoader to ensure each PDF is broken
    down into individual pages for citation."""
    
    collection = get_collection(native_db)

    print_with_time('Loading PDFs')
    files = os.listdir(DOCUMENT_SOURCE_DIRECTORY)
    
    for file in files:
        if file.endswith('.pdf'):
            loader = PyPDFLoader(f'{DOCUMENT_SOURCE_DIRECTORY}/{file}')
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = text_splitter.split_documents(pages)
            
            print(f'{file} - Pages: {len(pages)}')
            
            for index, chunk in enumerate(chunks):
                collection.upsert(
                    ids=[chunk.metadata.get("source") + str(index)], metadatas=chunk.metadata,
                    documents=chunk.page_content
                )


def get_collection(native_db) -> Collection:
    with SuppressStdout():
        print("DEBUG: call get_collection()")
        collection = None
        try:
            # Delete all documents
            native_db.delete_collection("books")
        except:
            pass
        finally:
            collection: Collection = native_db.get_or_create_collection("books", embedding_function=MyEmbeddingFunction())
        return collection
        
        
def format_docs(docs):
    """A simple document formatter if the type of document was not chunked"""
    return "\n".join(doc.page_content for doc in docs)

def main():
    
    llm = OllamaLLM(model="llama3.2")
        
    native_db = chromadb.PersistentClient("./data_store")
    
    read_files(native_db)
    
    db = Chroma(client=native_db, collection_name="books", embedding_function=embeddingModel)
      
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )

    # Initialize the primary chain outside the Q&A while loop.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 100, 'lambda_mult': 0.25} # Get many documents back, we do 100 because we're dealing with books
    )

    # Start the REPL
    while True:
        query = input("\nQuery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # Call the QA chain to print the response
        resp = rag_chain.invoke({"input": query})

        # Here, we'll print the entire response object, but normally you would only deal with the
        # answer: resp["answer"]
        print(resp)