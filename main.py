import streamlit as st
import time
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from langchain.chains import RetrievalQAWithSourcesChain

from dotenv import load_dotenv
load_dotenv()

file_path="vector_index.pkl"

llm = OpenAI(model_name= 'gpt-3.5-turbo-instruct' ,temperature=0.9, max_tokens=500) 
st.title('News Research tool')


def simulate_processing():
    # Simulate processing by sleeping for a few seconds
    for i in range(1, 6):
        st.write(f"Processing... {i}/5")
        time.sleep(1)
    st.success("Processing completed!")

num_inputs = 3
urls = []
# Define the sidebar inputs using a for loop
with st.sidebar:
    st.title("News Article URL's")
    for i in range(num_inputs):
        input_label = f"URL{i+1}"  # Customize the input heading
        input_value = st.text_input(input_label)
        urls.append(input_value)
    
    button_clicked = st.button("Process")


if button_clicked:
    loader = UnstructuredURLLoader( urls = urls)
    data = loader.load()
    r_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
    chunk_size = 1000,  # size of each chunk created
    chunk_overlap  = 0,  # size of  overlap between chunks in order to maintain the context
    length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
)
    chunks = r_splitter.split_documents(data)
    # Create the embeddings of the chunks using openAIEmbeddings
    embeddings = OpenAIEmbeddings()
    simulate_processing()
# Pass the documents and embeddings inorder to create FAISS vector index
    vectorindex_openai = FAISS.from_documents(chunks, embeddings)

    vectorindex_openai.save_local("faiss_store")
    
query = st.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        #with open(file_path, "rb") as f:
            #vectorstore = pickle.load(f)
            x = FAISS.load_local("faiss_store", OpenAIEmbeddings())
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=x.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)

