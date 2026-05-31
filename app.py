import streamlit as st
import re
import uuid
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

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
        "ignore previous", 
        "ignore all", 
        "system prompt", 
        "jailbreak", 
        "ignore as instruções", 
        "esqueça o que foi dito",
        "esqueça as regras"
    ]
    return any(item in pergunta.lower() for item in blacklist)


def masquerar_dados(texto):
    """
    Mecanismo de Sanitização e Proteção de Privacidade (Regex) - Capítulo 2.3.2 do TCC.
    Identifica e ofusca PII (Dados Pessoais Identificáveis) antes de persistir em logs.
    """
    # Mascarar CPF no formato XXX.XXX.XXX-XX ou sequências de 11 dígitos
    texto = re.sub(r"\b\d{3}\.\d{3}\.\d{3}\-\d{2}\b", "***.***.***-**", texto)
    texto = re.sub(r"\b\d{11}\b", "***********", texto)
    
    # Mascarar CNPJ no formato XX.XXX.XXX/XXXX-XX
    texto = re.sub(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}\-\d{2}\b", "**.***.***/****-**", texto)
    return texto

# ==============================================================================
# 🏗️ PIPELINE RAG (INGESTÃO E PROCESSAMENTO LOCAL)
# ==============================================================================

@st.cache_resource
def inicializar_banco_conhecimento():
    """
    Carrega os documentos locais das pastas corporativas, realiza a fragmentação
    semântica e persiste a indexação vetorial através do ChromaDB.
    """
    try:
        # Carga dos Manuais (docs/) e Perguntas Frequentes (faq/)
        loader_pdf = PyPDFDirectoryLoader("docs/")
        loader_faq = DirectoryLoader("faq/")
        
        documentos = loader_pdf.load() + loader_faq.load()
        
        # Fragmentação Semântica (Chunking de 800 caracteres com overlap de 150)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        fragmentos = splitter.split_documents(documentos)
        
        # Modelo de Embeddings Multilíngue Local via HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        
        # Banco Vetorial ChromaDB com persistência no diretório local ./db
        vectorstore = Chroma.from_documents(
            documents=fragmentos, 
            embedding=embeddings, 
            persist_directory="./db"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao inicializar base de conhecimento: {e}")
        return None

# Inicializar o banco estruturado
vectorstore = inicializar_banco_conhecimento()

if vectorstore:
    # Recuperador configurado para extrair as 6 partições mais semelhantes (k=6)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    
    # Modelo Cognitivo Local Llama 3.1 hospedado via servidor Ollama (Temp = 0.1)
    llm = OllamaLLM(model="llama3.1:8b", temperature=0.1)
    
    # Engenharia de Prompts - Retenção e Contextualização do Histórico
    contextualize_q_system_prompt = (
        "Dado um histórico de conversa e a última pergunta do usuário "
        "que pode fazer referência ao contexto do histórico, formule uma pergunta "
        "independente que possa ser entendida sem o histórico de conversa. "
        "NÃO responda à pergunta, apenas reformule-a se necessário, caso contrário, retorne-a como está."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
   # Prompt do Sistema Administrador - Restrição de Escopo de Atendimento
    system_prompt = """Você é o assistente virtual inteligente e seguro da Yellon Sig Seguros.
Use estritamente os seguintes pedaços de contexto recuperados para responder à pergunta.
Se você não sabe a resposta ou se ela não estiver presente no contexto fornecido, diga de forma educada
que não possui essa informação no momento e oriente o cliente a ligar para a Central de Atendimento
pelos telefones 4004-5423 (Capitais) ou 0800-709-5423 (Demais localidades).
Mantenha as respostas objetivas, profissionais e estritamente baseadas nos fatos fornecidos.

Contexto recuperado:
{context}"""