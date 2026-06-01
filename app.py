import streamlit as st
import re
import uuid
import time
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Configuração da Interface
st.set_page_config(page_title="Yellon Sig Seguros", page_icon="🛡️", layout="centered")
st.title("🛡️ Assistente Virtual - Yellon Sig Seguros")
st.markdown("### Atendimento inteligente ao segurado")

def detectar_prompt_injection(pergunta):
    blacklist = ["ignore previous", "ignore all", "system prompt", "jailbreak", "ignore as instruções"]
    return any(item in pergunta.lower() for item in blacklist)

def masquerar_dados(texto):
    texto = re.sub(r"\b\d{3}\.\d{3}\.\d{3}\-\d{2}\b", "***.***.***-**", texto)
    texto = re.sub(r"\b\d{11}\b", "***********", texto)
    return texto

@st.cache_resource
def inicializar_banco_conhecimento():
    try:
        documentos = []
        if os.path.exists("docs/"):
            loader_pdf = PyPDFDirectoryLoader("docs/")
            documentos.extend(loader_pdf.load())
        if os.path.exists("faq/"):
            for arquivo in os.listdir("faq/"):
                if arquivo.endswith(".txt"):
                    loader_txt = TextLoader(os.path.join("faq/", arquivo), encoding="utf-8")
                    documentos.extend(loader_txt.load())
        
        if not documentos:
            st.error("Nenhum documento encontrado nas pastas docs/ ou faq/")
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        fragmentos = splitter.split_documents(documentos)
        
        # Modelo de Embeddings leve e multilíngue
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory="./db")
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao inicializar base de conhecimento: {e}")
        return None

vectorstore = inicializar_banco_conhecimento()

if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # LLM via API pública externa (Roda na nuvem, zero consumo de RAM no Streamlit)
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={"max_new_tokens": 512, "temperature": 0.1}
    )
    
    contextualize_q_system_prompt = (
        "Dado um histórico de conversa e a última pergunta do usuário, "
        "formule uma pergunta independente que possa ser entendida sem o histórico."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    system_prompt = """Você é o assistente virtual da Yellon Sig Seguros. 
Use estritamente o contexto para responder. Se não souber, oriente a ligar para 4004-5423 ou 0800-709-5423.

Contexto:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pergunta_usuario := st.chat_input("Digite sua dúvida sobre seguros..."):
    if len(pergunta_usuario) > 800 or detectar_prompt_injection(pergunta_usuario):
        st.error("Solicitação inválida ou insegura.")
        st.stop()

    pergunta_higienizada = masquerar_dados(pergunta_usuario)
    st.session_state.messages.append({"role": "user", "content": pergunta_higienizada})
    with st.chat_message("user"):
        st.markdown(pergunta_higienizada)

    if vectorstore:
        with st.chat_message("assistant"):
            mensagem_placeholder = st.empty()
            mensagem_placeholder.markdown("🔍 *Consultando base de conhecimento segura...*")
            try:
                resposta_objeto = rag_chain.invoke({
                    "input": pergunta_higienizada, 
                    "chat_history": st.session_state.chat_history
                })
                resposta_texto = resposta_objeto["answer"]
                
                mensagem_placeholder.markdown(resposta_texto)
                st.session_state.messages.append({"role": "assistant", "content": resposta_texto})
                st.session_state.chat_history.extend([
                    HumanMessage(content=pergunta_higienizada),
                    AIMessage(content=resposta_texto)
                ])
                st.rerun()
            except Exception as e:
                st.error(f"Erro de processamento: {e}")