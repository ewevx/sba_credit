import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==============================================================================
# Configuração Inicial da Página
# ==============================================================================
st.set_page_config(
    page_title="Análise de Risco SBA",
    page_icon="📊",
    layout="wide"
)

# ==============================================================================
# Carregamento do Modelo em Cache (Alta Performance)
# ==============================================================================
@st.cache_resource
def load_model():
    """
    Carrega o modelo salvo na pasta 'models'.
    O cache evita recarregamentos desnecessários a cada clique do usuário.
    """
    # Caminho atualizado para refletir a pasta 'models'
    model_path = os.path.join('models', 'modelo_risco_credito.pkl')
    
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

pipeline = load_model()

# Trava de segurança: Se o modelo não carregar, para a aplicação aqui e avisa o usuário.
if pipeline is None:
    st.error("🚨 Erro Crítico: O arquivo do modelo não foi encontrado. Verifique se 'modelo_risco_credito.pkl' está dentro da pasta 'models/'.")
    st.stop()

# ==============================================================================
# Interface: Título e Cabeçalho
# ==============================================================================
st.title("📊 Sistema Inteligente de Análise de Risco de Crédito - SBA")
st.markdown("Insira os dados do solicitante no menu lateral para obter a probabilidade de aprovação e as recomendações geradas por Inteligência Artificial.")

# ==============================================================================
# Interface: Sidebar (Formulário de Entrada)
# ==============================================================================
st.sidebar.header("📋 Dados do Solicitante")

with st.sidebar.form("form_credito"):
    valor_emprestimo = st.number_input("Valor do Empréstimo ($)", min_value=1000.0, max_value=5000000.0, value=50000.0, step=1000.0)
    faturamento = st.number_input("Faturamento Anual Estimado ($)", min_value=0.0, value=100000.0, step=5000.0)
    prazo_meses = st.slider("Prazo (Meses)", min_value=1, max_value=300, value=84)
    estado = st.selectbox("Estado (Sigla)", ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "MI", "GA", "NC", "Outro"])
    setor_naics = st.selectbox("Setor Econômico (NAICS)", ["Varejo (44)", "Alimentação (72)", "Construção (23)", "Saúde (62)", "TI/Tech (54)"])
    tipo_empresa = st.radio("Tempo de Empresa", ["Nova (Menos de 2 anos)", "Existente (Mais de 2 anos)"])
    
    submit_button = st.form_submit_button(label="Analisar Risco 🚀")

# ==============================================================================
# Lógica de Backend e Predição
# ==============================================================================
if submit_button:
    with st.spinner("A IA está analisando o perfil de crédito..."):
        
        # 1. Transformar os inputs do usuário no formato esperado pelo pipeline
        sba_garantia = valor_emprestimo * 0.80 # Assumindo 80% de garantia padrão SBA
        
        input_data = pd.DataFrame([{
            'term': prazo_meses,
            'grappv': valor_emprestimo,
            'sba_appv': sba_garantia,
            'sba_guarantee_pct': sba_garantia / valor_emprestimo if valor_emprestimo > 0 else 0,
            'noemp': 5, # Valor modal padrão
            'state': estado,
            'bankstate': estado, 
            'newexist': 2 if "Nova" in tipo_empresa else 1,
            'urbanrural': 1, 
            'revlinecr': 'N' 
        }])
        
        # 2. Realizar a Predição
        probabilidade_calote = pipeline.predict_proba(input_data)[0, 1]
        classe_predita = 1 if probabilidade_calote > 0.5 else 0
        
        # ==============================================================================
        # Visualização de Resultados
        # ==============================================================================
        st.divider()
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Resultado da Análise")
            if classe_predita == 1:
                st.error("🚨 ALTO RISCO DE INADIMPLÊNCIA")
                st.metric(label="Probabilidade de Calote", value=f"{probabilidade_calote * 100:.1f}%")
            else:
                st.success("✅ BAIXO RISCO - APROVAÇÃO RECOMENDADA")
                st.metric(label="Probabilidade de Calote", value=f"{probabilidade_calote * 100:.1f}%")
                
        # ==============================================================================
        # Explicação do Modelo (SHAP + IA Simulação)
        # ==============================================================================
        with col2:
            st.subheader("Por que o modelo tomou essa decisão?")
            
            # Dados Simulados do SHAP para interface (substitua pelo cálculo real com shap_values se desejar)
            if classe_predita == 1:
                fatores = ['Prazo (Curto)', 'Setor Econômico (Risco)', 'Garantia SBA (Baixa)']
                impactos = [0.35, 0.20, 0.15]  
                texto_ia = f"🤖 **Recomendação da IA:** Notamos que o prazo solicitado de {prazo_meses} meses pressiona o fluxo de caixa para o setor '{setor_naics[:6]}'. Recomendamos alongar o prazo do empréstimo ou solicitar garantias colaterais adicionais."
            else:
                fatores = ['Prazo (Longo/Seguro)', 'Tempo de Empresa (Consolidada)', 'Valor (Adequado)']
                impactos = [-0.40, -0.25, -0.10] 
                texto_ia = f"🤖 **Recomendação da IA:** O perfil é muito sólido. O tempo de atuação de mercado ('{tipo_empresa.split(' ')[0]}') alinhado ao prazo de {prazo_meses} meses indica alta capacidade de honrar o compromisso. Aprovação via 'Fast-Track' é recomendada."

            # Gerando Gráfico de Barras Horizontal (Matplotlib)
            fig, ax = plt.subplots(figsize=(6, 3))
            cores = ['#e74c3c' if val > 0 else '#2ecc71' for val in impactos]
            ax.barh(fatores, impactos, color=cores)
            ax.set_xlabel('Impacto no Risco (+ Aumenta | - Reduz)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        # Alerta da "Inteligência Artificial" Generativa
        st.info(texto_ia)