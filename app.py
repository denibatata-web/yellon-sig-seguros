import streamlit as st
import re
import uuid
import time
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
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
    Carrega os documentos locais das pastas corporativas de forma leve,
    realiza a fragmentação semântica e persiste a indexação vetorial.
    """
    try:
        documentos = []
        
        # 1. Carrega os PDFs da pasta docs/
        if os.path.exists("docs/"):
            loader_pdf = PyPDFDirectoryLoader("docs/")
            documentos.extend(loader_pdf.load())
            
        # 2. Carrega arquivos de texto (.txt) da pasta faq/ nativamente sem unstructured
        if os.path.exists("faq/"):
            for arquivo in os.listdir("faq/"):
                if arquivo.endswith(".txt"):
                    caminho_completo = os.path.join("faq/", arquivo)
                    loader_txt = TextLoader(caminho_completo, encoding="utf-8")
                    documentos.extend(loader_txt.load())
        
        if not documentos:
            st.error("Nenhum documento encontrado nas pastas docs/ ou faq/")
            return None

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
    
    # 🤗 LLM Público: Modelo estável que roda direto na nuvem sem precisar de chaves/tokens
    llm = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.1}
    )
    
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
    
    # Prompt do Sistema Administrador - Restrição de Escopo utilizando Aspas Triplas Legítminas
    system_prompt = """Você é o assistente virtual inteligente e seguro da Yellon Sig Seguros.
Use estritamente os seguintes pedaços de contexto recuperados para responder à pergunta.
Se você não sabe a resposta ou se ela não estiver presente no contexto fornecido, diga de forma educada
que não possui essa informação no momento e oriente o cliente a ligar para a Central de Atendimento
pelos telefones 4004-5423 (Capitais) ou 0800-709-5423 (Demais localidades).
Mantenha as respostas objetivas, professionals e estritamente baseadas nos fatos fornecidos.

Contexto recuperado:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Consolidação da Cadeia RAG Completa
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ==============================================================================
# 🧠 GERENCIAMENTO DE ESTADO E FLUXO DA CONVERSA
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Renderização do Histórico Visual no Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================================================================
# 💬 INTERAÇÃO COMPUTACIONAL EM TEMPO DE EXECUÇÃO
# ==============================================================================
if pergunta_usuario := st.chat_input("Digite sua dúvida sobre seguros..."):
    
    # 🔍 TESTE DE SEGURANÇA 1: Validação de Negação de Serviço por Volumetria (DoS)
    if len(pergunta_usuario) > 800:
        st.error("❌ Erro: O tamanho do payload excede o limite seguro de caracteres da aplicação.")
        st.stop()
        
    # 🔍 TESTE DE SEGURANÇA 2: Validação de Tentativa de Prompt Injection
    if detectar_prompt_injection(pergunta_usuario):
        st.error("⚠️ Solicitação bloqueada por segurança. Padrão adversário detectado.")
        st.stop()

    # 🔍 TESTE DE SEGURANÇA 3: Higienização de Dados Pessoais (Mascaramento de PII)
    pergunta_higienizada = masquerar_dados(pergunta_usuario)
    
    # Registrar entrada higienizada na tela
    st.session_state.messages.append({"role": "user", "content": pergunta_higienizada})
    with st.chat_message("user"):
        st.markdown(pergunta_higienizada)

    # Geração de Resposta via Pipeline RAG
    if vectorstore:
        with st.chat_message("assistant"):
            mensagem_placeholder = st.empty()
            mensagem_placeholder.markdown("🔍 *Consultando base de conhecimento segura...*")
            
            try:
                # Execução da Cadeia RAG
                resposta_objeto = rag_chain.invoke({
                    "input": pergunta_higienizada, 
                    "chat_history": st.session_state.chat_history
                })
                resposta_texto = resposta_objeto["answer"]
                
                # Exibição progressiva (Efeito de digitação)
                resposta_gradual = ""
                for caractere in resposta_texto:
                    resposta_gradual += caractere
                    mensagem_placeholder.markdown(resposta_gradual + "▌")
                    time.sleep(0.005)
                mensagem_placeholder.markdown(resposta_texto)
                
                # Identificador Único de Mensagem
                msg_id = str(uuid.uuid4())
                st.session_state.messages.append({"role": "assistant", "content": resposta_texto, "id": msg_id})
                
                # Atualização do histórico de contexto do LangChain
                st.session_state.chat_history.extend([
                    HumanMessage(content=pergunta_higienizada),
                    AIMessage(content=resposta_texto)
                ])
                st.rerun()
                
            except Exception as e:
                st.error(f"Desculpe, ocorreu um erro de processamento cognitivo local: {e}")
    else:
        st.error("O sistema está indisponível pois o banco vetorial não foi carregado corretamente.")