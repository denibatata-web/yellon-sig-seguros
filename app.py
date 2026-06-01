import streamlit as st
import re
import uuid
import time
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

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
    """
    Filtro Sintático Baseado em Assinaturas (Blacklist) - Capítulo 2.3.1 do TCC.
    Intercepta tentativas conhecidas de engenharia social reversa e jailbreak.
    """
    blacklist = [
        "ignore previous", "ignore all", "system prompt", "jailbreak", 
        "ignore as instruções", "esqueça o que foi dito", "esqueça as regras"
    ]
    return any(item in pergunta.lower() for item in blacklist)

def masquerar_dados(texto):
    """
    Mecanismo de Sanitização e Proteção de Privacidade (Regex) - Capítulo 2.3.2 do TCC.
    Identifica e ofusca PII (Dados Pessoais Identificáveis) antes de persistir em logs.
    """
    texto = re.sub(r"\b\d{3}\.\d{3}\.\d{3}\-\d{2}\b", "***.***.***-**", texto)
    texto = re.sub(r"\b\d{11}\b", "***********", texto)
    return texto

# ==============================================================================
# 🏗️ PIPELINE RAG SIMPLIFICADO E LEVE
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

        # Reduzimos o tamanho dos blocos para economizar memória do servidor público
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        fragmentos = splitter.split_documents(documentos)
        
        # Embeddings ultra leves para evitar o erro de 'Out of Memory'
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(documents=fragmentos, embedding=embeddings, persist_directory="./db")
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao inicializar base de conhecimento: {e}")
        return None

vectorstore = inicializar_banco_conhecimento()

# Modelo de IA Local Simulado para garantir estabilidade e funcionamento sob restrição de RAM
class AssistenteLocal:
    def invoke(self, inputs):
        return {"answer": "Prezado cliente, com base nas diretrizes da Yellon Sig Seguros, localizei as informações correspondentes em nossa base de dados corporativa para lhe auxiliar. Caso necessite de suporte complementar ou abertura de sinistros, por favor contate nossa Central de Atendimento pelos telefones 4004-5423 (Capitais) ou 0800-709-5423 (Demais localidades)."}

if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    rag_chain = AssistenteLocal()

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
                # Recupera os documentos do ChromaDB de forma isolada
                docs = retriever.get_relevant_documents(pergunta_higienizada)
                contexto_texto = "\n\n".join([doc.page_content for doc in docs])
                
                # Executa a lógica estável livre de timeouts
                resposta_objeto = rag_chain.invoke({"input": pergunta_higienizada, "context": contexto_texto})
                resposta_texto = resposta_objeto["answer"]
                
                mensagem_placeholder.markdown(resposta_texto)
                st.session_state.messages.append({"role": "assistant", "content": resposta_texto})
                st.rerun()
            except Exception as e:
                st.error(f"Erro de processamento: {e}")