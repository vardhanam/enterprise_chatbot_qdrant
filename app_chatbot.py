from transformers import (
AutoTokenizer,
AutoModelForCausalLM,
BitsAndBytesConfig,
pipeline
)

import transformers
import torch
import streamlit as st

from langchain.llms import HuggingFacePipeline

import os
import time  # Just for simulating a delay

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

#function to load the llama2 llm()

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

UPLOAD_DIR = '/home/vardhanam/enterprise_chatbot/uploaded_pdfs'

def save_uploaded_file(uploaded_file):
    try:
        # Create a directory to save the file if it doesn't exist


        # Save the file
        with open(os.path.join(UPLOAD_DIR, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())

        return True

    except Exception as e:
        # If there's an error, print the exception
        print(e)
        return False

def generate_response(query):
     return chain.invoke(query)


@st.cache_resource
def load_llm():

    #Loading the Llama-2 Model
    model_name='mistralai/Mistral-7B-Instruct-v0.2'
    model_config = transformers.AutoConfig.from_pretrained(
    model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Activate 4-bit precision base model loading
    use_4bit = True
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    #################################################################
    # Set up quantization config
    #################################################################
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
    )
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)


    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    )

    # Building a LLM text-generation pipeline
    text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
    )

    llm = HuggingFacePipeline(pipeline= text_generation_pipeline)

    return llm


@st.cache_resource()
def process_document(folder_name):


    global text_splitter
    # Simulate some document processing delay
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )
    loader = DirectoryLoader(folder_name, loader_cls=PyPDFLoader)
    docs = loader.load_and_split(text_splitter=text_splitter)

    #Loading the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    global qdrant_vectorstore
    qdrant_vectorstore = Qdrant.from_documents(
        docs,
        embeddings,
        location = ":memory:",
        collection_name = "depp_heard_transcripts",
    )

    qdrant_retriever = qdrant_vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    global chain
    chain = (
    {"context": qdrant_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    return chain



with st.spinner("Loading llm"):
    llm = load_llm()


with st.spinner("Creating Vector DB"):
    chain = process_document(UPLOAD_DIR)


with open('/home/vardhanam/enterprise_chatbot/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

authenticator.login()


if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    # Streamlit app starts here
    st.title('Legal Documents Processing App')

    with st.form("Upload Form", clear_on_submit= True):

        # Use st.file_uploader to upload multiple files
        uploaded_files = st.file_uploader("Upload Legal Document PDF files:", type='pdf', accept_multiple_files=True)

        submitted = st.form_submit_button("Submit")

        if submitted:
            # If files were uploaded, iterate over the list of uploaded files
            if uploaded_files is not None:
                for uploaded_file in uploaded_files:
                    # Save each uploaded file to disk
                    if save_uploaded_file(uploaded_file):
                        st.success(f"'{uploaded_file.name}' saved successfully!")

                    else:
                        st.error(f"Failed to save '{uploaded_file.name}'")
                with st.spinner("Refreshing Vector DB"):
                    process_document.clear()
                    chain = process_document(UPLOAD_DIR)
                    uploaded_files = None


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing Query"):
                stream = generate_response(prompt)
                st.markdown(stream)

        st.session_state.messages.append({"role": "assistant", "content": stream})

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')