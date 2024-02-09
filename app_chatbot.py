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
#funtcion to load the llama2 llm()


def load_llm():

    #Loading the Llama-2 Model
    model_name='NousResearch/Llama-2-7b-chat-hf'
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
    max_new_tokens=300,
    )

    llm = HuggingFacePipeline(pipeline= text_generation_pipeline)

    return llm

if 'llm_loaded' not in st.session_state:
    st.session_state.llm_loaded = False

if 'llm_loaded' in st.session_state and not st.session_state.llm_loaded:
    with st.spinner('Loading the language model...'):
        if 'llm' not in st.session_state:
            st.session_state.llm = load_llm()
    st.session_state.llm_loaded = True

import os
import time  # Just for simulating a delay

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def process_document(folder_name, llm):

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

    chain = (
    {"context": qdrant_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    return chain


# Streamlit app starts here
st.title('Document Processing App')

folder_path = st.text_area("Enter folder path:", max_chars=50, height=5, help="Press the submit button below to submit")

# Button to explicitly submit the input
if st.button("Submit", key="folder_path_button"):

    if folder_path:
        if os.path.isdir(folder_path):
            with st.spinner(f'Processing documents in: {folder_path}...'):
                st.session_state.chain = process_document(folder_path, st.session_state.llm)
        else:
            st.error('Folder not found. Please enter a valid folder name.')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ''

query = st.text_area("Enter your query here:", max_chars=200, height=5, help="Press the submit button below to submit")

if query:

    if st.button("Submit", key= "query_button"):
        with st.spinner('Analyzing query...'):
            response = st.session_state.chain.invoke(query)
        st.session_state.chat_history += f"> {query}\n{response}\n\n"

    # Display conversation history
    st.text_area("Conversation:", st.session_state.chat_history, height=1000, key="conversation_area")