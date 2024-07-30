import logging
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import ScaNN
from sentence_transformers import SentenceTransformer
import base64
import google.generativeai as gemini_client
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
from dotenv import load_dotenv

load_dotenv()
g_api_key = os.getenv("GOOGLE_API_KEY")
q_api_key = os.getenv("QDRANT_API_KEY")

logging.basicConfig(level=logging.INFO)

# Inicializando o LLM do Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=g_api_key)

# Inicializando o cliente Qdrant
collection_name = "db_rag_metric"
search_client = QdrantClient(
    url="https://d58764bf-41d0-4544-8b0d-87ffc570331d.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=q_api_key,
)

# Inicializando os embeddings necessários
embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# Criação do prompt
prompt_template = "Seu nome é Ramin e você é uma Engenheira de Software, \
responda a pergunta {question} com base nos dados {result_qdrant} e explique, \
de forma humanizada os pontos relacionados aos dados retornados, se possível, forneça uma tabela com os dados de resultados ao final e se possível, como aplicar as métricas."

prompt = PromptTemplate(
    input_variables=["question", "result_qdrant"],
    template=prompt_template
)
chain_1 = LLMChain(llm=llm, prompt=prompt)

# Streamlit Page Configuration
st.set_page_config(
    page_title="RAMIN - Retrieval-Augmented of metrics information for microservices",
    page_icon="img/bot3.jpeg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/gomesrocha/ramim",
        "Report a bug": "https://github.com/gomesrocha/ramim",
        "About": """
            ## RAMIN - Retrieval-Augmented of metrics information for microservices
            
            RAMIN is a RAG system for microservice metrics
        """
    }
)

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    """
    Display Streamlit updates and handle the chat interface.
    """

    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330033,
                0 0 10px #660066,
                0 0 15px #990099,
                0 0 20px #CC00CC,
                0 0 25px #FF00FF,
                0 0 30px #FF33FF,
                0 0 35px #FF66FF;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load and display sidebar image with glowing effect
    img_path = "img/bot3.jpeg"
    img_base64 = img_to_base64(img_path)
    st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    # Collection selection
    collection_mapping = {
        "Microservices Metrics": "db_rag_metric",
    }

    # Collection selection
    selected_collection = st.sidebar.selectbox(
        "Escolha a coleção de dados",
        list(collection_mapping.keys())
    )

    collection_name = collection_mapping[selected_collection]

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        ### Como interagir com a RAMIN
        - **Pergunte sobre Métricas para microsserviços **: Digite suas perguntas sobre as métricas de microsserviços e como usar.
        
        """)
    st.sidebar.markdown("---")

    # Interface Streamlit
    st.image("img/bot3.jpeg", width=50)  # Displaying the image as a thumbnail above the text input

    question = st.text_input("Pergunte a RAMIN:")

    if st.button("Enviar"):
        if question:
            # Desabilitar a barra de progresso
            query_embedding = embeddings.encode([question], show_progress_bar=False)[0].tolist()
            search_results = search_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=10,
            )
            st.write(question)
            st.write(search_results)
            inputs = {"question": question, "result_qdrant": search_results}
            result = chain_1.run(inputs)
            st.write(result)
        else:
            st.write("Pergunte a Ramin")

if __name__ == "__main__":
    main()
