
# Legal Documents Processing App with Chatbot

This application combines the power of Large Language Model with the ease of use of Streamlit to offer an innovative solution for interacting with legal documents. Users can upload PDFs of legal documents, which the app processes to create a searchable vector database. Additionally, the app features an interactive chatbot that can answer queries based on the processed documents. This app can be used by enterprises for querying on their internal documents.

## Features

- **PDF Document Upload:** Users can upload multiple PDF files, which are then processed and stored for querying.
- **Vector Database:** Utilizes `sentence-transformers/all-mpnet-base-v2` for embeddings and Qdrant for creating a searchable vector database from the uploaded documents.
- **Interactive Chat:** Powered by the `NousResearch/Llama-2-7b-chat-hf` model, the chatbot provides answers to user queries by referencing the content of the uploaded documents.
- **User Authentication:** Incorporates a simple authentication system to ensure that only authorized users can access the app.

## Getting Started

### Prerequisites

- Python 3.6 or later
- Pip package manager

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/vardhanam/enterprise_chatbot_qdrant.git
cd legal-docs-processing-app
pip install -r requirements.txt
```

### Running the Application

To run the app, navigate to the project directory and execute:

```bash
streamlit run app.py
```

Replace `app.py` with the path to your Streamlit application script if necessary.

## Usage

Once the application is running, navigate to the provided URL in your web browser:

1. Log in with your credentials. All the credentials are stored in the config.yaml file. If you want to add or remove credentials, you can do so by manipulating the config.yaml file.



2. Use the "Upload Legal Document PDF files" section to upload your PDF documents.


3. Interact with the chatbot through the "What is up?" chat input to query the processed documents.

## Contributing

We welcome contributions to improve the app or extend its capabilities. Please feel free to fork the repository, make your changes, and submit a pull request.


## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the incredible language models and the `transformers` library.
- [Streamlit](https://streamlit.io/) for the intuitive app development framework.
- [Qdrant](https://qdrant.tech/) for the vector storage solution.

## Jupyter Notebook

- The repository also contains a Jupyter Notebook by the name app_chatbot.ipynb. You can execute the cells of the notebook in a step by step fashion to launch a gradio app. Don't forget to install the requirements file like before.
- The gradio app has two tabs. One for uploading new documents. And the other is for querying the documents.