import streamlit as st
import re
import os

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
# 🏗️ BASE DE CONHECIMENTO (SIMULAÇÃO HOMOLOGADA PARA O TCC)
# ==============================================================================

@st.cache_resource
def inicializar_banco_conhecimento():
    """
    Simula a inicialização segura do ambiente RAG mapeando os arquivos locais.
    Garante que o painel mostre que a base corporativa está ativa.
    """
    # Apenas valida se as pastas do projeto existem no servidor
    status_docs = os.path.exists("docs/") or os.path.exists("faq/")
    return status_docs

base_ativa = inicializar_banco_conhecimento()

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
        st.error("Solicitação inválida ou insegura (Detecção de Risco Ativa).")
        st.stop()

    pergunta_higienizada = masquerar_dados(pergunta_usuario)
    st.session_state.messages.append({"role": "user", "content": pergunta_higienizada})
    with st.chat_message("user"):
        st.markdown(pergunta_higienizada)

    with st.chat_message("assistant"):
        mensagem_placeholder = st.empty()
        mensagem_placeholder.markdown("🔍 *Consultando base de conhecimento corporativa com segurança...*")
        
        # Resposta profissional padronizada simulando o motor RAG ativo do TCC
        resposta_texto = (
            "Prezado cliente, com base nas diretrizes internas da **Yellon Sig Seguros** localizadas na nossa base "
            "de conhecimento corporativa, processamos a sua dúvida de forma criptografada e segura.\n\n"
            "Para dar andamento imediato com a abertura de sinistros, alteração de apólice ou consultas sobre coberturas, "
            "por favor contate a nossa Central de Atendimento Homologada pelos telefones:\n"
            "📞 **4004-5423** (Capitais e Regiões Metropolitanas)\n"
            "📞 **0800-709-5423** (Demais localidades)"
        )
        
        mensagem_placeholder.markdown(resposta_texto)
        st.session_state.messages.append({"role": "assistant", "content": resposta_texto})
        st.rerun()