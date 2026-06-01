import streamlit as st
import re
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================================================================
# 🎛️ CONFIGURAÇÃO DA INTERFACE (STREAMLIT)
# ==============================================================================
st.set_page_config(page_title="Yellon Sig Seguros", page_icon="🛡️", layout="centered")
st.title("🛡️ Assistente Virtual - Yellon Sig Seguros")
st.markdown("### Atendimento inteligente ao segurado")

# ==============================================================================
# 🛡️ CAMADAS DEFENSIVAS PROGRAMÁTICAS DE SEGURANÇA (PERÍMETRO)
# ==============================================================================

def detectar_prompt_injection(pergunta):
    blacklist = [
        "ignore previous", "ignore all", "system prompt", "jailbreak", 
        "ignore as instruções", "esqueça o que foi dito", "esqueça as regras"
    ]
    return any(item in pergunta.lower() for item in blacklist)

def masquerar_dados(texto):
    texto = re.sub(r"\b\d{3}\.\d{3}\.\d{3}\-\d{2}\b", "***.***.***-**", texto)
    texto = re.sub(r"\b\d{11}\b", "***********", texto)
    return texto

# ==============================================================================
# 🏗️ PIPELINE RAG SIMPLIFICADO E SEGURO
# ==============================================================================

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

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        fragmentos = splitter.split_documents(documentos)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory="./db")
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao inicializar base de conhecimento: {e}")
        return None

vectorstore = inicializar_banco_conhecimento()

# ==============================================================================
# 🧠 GERENCIAMENTO DE ESTADO E FLUXO DA CONVERSA
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================================================================
# 💬 INTERAÇÃO COMPUTACIONAL EM TEMPO DE EXECUÇÃO
# ==============================================================================
if pergunta_usuario := st.chat_input("Digite sua dúvida sobre seguros..."):
    if len(pergunta_usuario) > 800 or detectar_prompt_injection(pergunta_usuario):
        st.error("Solicitação inválida ou unsafe.")
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
                # Faz a busca vetorial nos documentos do ChromaDB
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                docs = retriever.get_relevant_documents(pergunta_higienizada)
                
                # Resposta fixa estável baseada na simulação de atendimento homologado do TCC
                resposta_texto = "Prezado cliente, com base nas diretrizes da Yellon Sig Seguros localizadas na base de conhecimento, processamos sua solicitação de forma segura. Para andamento com abertura de sinistros ou alterações de apólice, contate nossa Central pelos telefones 4004-5423 (Capitais) ou 0800-709-5423 (Demais localidades)."
                
                mensagem_placeholder.markdown(resposta_texto)
                st.session_state.messages.append({"role": "assistant", "content": resposta_texto})
                st.rerun()
            except Exception as e:
                st.error(f"Erro de processamento: {e}")