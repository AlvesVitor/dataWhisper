import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import streamlit as st

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

st.set_page_config(
    page_title="🤖 Agente Autônomo analista de dados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EDAAgent:
    """Agente Autônomo para Análise Exploratória de Dados"""
    
    def __init__(self, df: pd.DataFrame, api_key: str, model_name: str):
        self.df = df
        self.conclusions = []
        self.analysis_history = []
        self.generated_plots = []
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=api_key
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.tools = self._setup_tools()
        
        self.agent = self._setup_agent()

    def _setup_tools(self) -> List[Tool]:
        """Configura as ferramentas disponíveis para o agente"""
        
        tools = [
            Tool(
                name="get_dataframe_info",
                func=self._get_dataframe_info,
                description="Retorna informações básicas sobre o dataframe: colunas, tipos, shape, etc."
            ),
            Tool(
                name="get_statistical_summary",
                func=self._get_statistical_summary,
                description="Retorna resumo estatístico das colunas numéricas do dataframe"
            ),
            Tool(
                name="check_missing_values",
                func=self._check_missing_values,
                description="Verifica valores faltantes no dataframe"
            ),
            Tool(
                name="get_correlation_matrix",
                func=self._get_correlation_matrix,
                description="Calcula matriz de correlação entre variáveis numéricas"
            ),
            Tool(
                name="detect_outliers",
                func=self._detect_outliers,
                description="Detecta outliers usando o método IQR. Input: nome da coluna numérica"
            ),
            Tool(
                name="create_histogram",
                func=self._create_histogram,
                description="Cria histograma para uma coluna. Input: nome da coluna"
            ),
            Tool(
                name="create_scatter_plot",
                func=self._create_scatter_plot,
                description="Cria gráfico de dispersão. Input: 'coluna_x,coluna_y'"
            ),
            Tool(
                name="create_box_plot",
                func=self._create_box_plot,
                description="Cria box plot para uma coluna. Input: nome da coluna"
            ),
            Tool(
                name="add_conclusion",
                func=self._add_conclusion,
                description="Adiciona uma conclusão à análise. Input: texto da conclusão"
            )
        ]
        
        return tools

    def _setup_agent(self) -> AgentExecutor:
        """Configura o agente ReAct"""
        
        template = """Responda à pergunta do usuário da melhor forma possível. Você tem acesso às seguintes ferramentas:

{tools}

Use este formato EXATAMENTE:

Question: a pergunta de entrada que você deve responder
Thought: você deve sempre pensar sobre o que fazer
Action: a ação a tomar, deve ser uma de [{tool_names}]
Action Input: o input da ação
Observation: o resultado da ação
... (este padrão Thought/Action/Action Input/Observation pode repetir)
Thought: Agora eu sei a resposta final
Final Answer: a resposta final à pergunta original

IMPORTANTE:
- Use NO MÁXIMO 3 ações por pergunta
- Seja direto e objetivo
- Depois de obter informações suficientes, vá direto para "Final Answer"
- NÃO repita ações já executadas

Comece agora!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template=template
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=6,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor

    
    def _get_dataframe_info(self, _: str = "") -> str:
        """Retorna informações básicas do dataframe"""
        info = f"""
Informações do Dataset:
- Shape: {self.df.shape[0]} linhas × {self.df.shape[1]} colunas
- Colunas: {list(self.df.columns)}
- Tipos de dados: {self.df.dtypes.to_dict()}
- Memória usada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""
        return info

    def _get_statistical_summary(self, _: str = "") -> str:
        """Retorna resumo estatístico"""
        return self.df.describe().to_string()

    def _check_missing_values(self, _: str = "") -> str:
        """Verifica valores faltantes"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        result = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        })
        result = result[result['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
        
        if len(result) == 0:
            return "Não há valores faltantes no dataset."
        
        return f"Valores faltantes:\n{result.to_string()}"

    def _get_correlation_matrix(self, _: str = "") -> str:
        """Calcula matriz de correlação"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return "Não há colunas numéricas suficientes para calcular correlações."
        
        corr_matrix = self.df[numeric_cols].corr()
        return f"Matriz de Correlação:\n{corr_matrix.to_string()}"

    def _detect_outliers(self, column: str) -> str:
        """Detecta outliers usando IQR"""
        column = column.strip()
        
        if column not in self.df.columns:
            return f"Coluna '{column}' não encontrada."
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return f"Coluna '{column}' não é numérica."
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        return f"""
Análise de Outliers para '{column}':
- Q1: {Q1:.2f}
- Q3: {Q3:.2f}
- IQR: {IQR:.2f}
- Limite inferior: {lower_bound:.2f}
- Limite superior: {upper_bound:.2f}
- Número de outliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)
"""

    def _create_histogram(self, column: str) -> str:
        """Cria histograma"""
        column = column.strip()
        
        if column not in self.df.columns:
            return f"Coluna '{column}' não encontrada."
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df[column].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title(f'Histograma: {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequência')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        self.generated_plots.append({
            'type': 'histogram',
            'column': column,
            'image': img_base64
        })
        
        return f"Histograma criado para a coluna '{column}'"

    def _create_scatter_plot(self, columns: str) -> str:
        """Cria gráfico de dispersão"""
        try:
            col_x, col_y = [c.strip() for c in columns.split(',')]
        except:
            return "Formato inválido. Use: 'coluna_x,coluna_y'"
        
        if col_x not in self.df.columns or col_y not in self.df.columns:
            return f"Uma ou ambas as colunas não foram encontradas."
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.df[col_x], self.df[col_y], alpha=0.5)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f'Scatter Plot: {col_x} vs {col_y}')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        self.generated_plots.append({
            'type': 'scatter',
            'columns': f"{col_x} vs {col_y}",
            'image': img_base64
        })
        
        return f"Scatter plot criado para '{col_x}' vs '{col_y}'"

    def _create_box_plot(self, column: str) -> str:
        """Cria box plot"""
        column = column.strip()
        
        if column not in self.df.columns:
            return f"Coluna '{column}' não encontrada."
        
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df.boxplot(column=[column], ax=ax)
        ax.set_title(f'Box Plot: {column}')
        ax.set_ylabel(column)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        self.generated_plots.append({
            'type': 'boxplot',
            'column': column,
            'image': img_base64
        })
        
        return f"Box plot criado para a coluna '{column}'"

    def _add_conclusion(self, conclusion: str) -> str:
        """Adiciona conclusão"""
        self.conclusions.append(conclusion)
        return f"Conclusão adicionada: {conclusion}"

    def analyze(self, query: str) -> Dict[str, Any]:
        """Executa análise baseada na query"""
        
        simple_response = self._try_simple_answer(query)
        if simple_response:
            return {
                'success': True,
                'response': simple_response,
                'plots': self.generated_plots.copy(),
                'conclusions': self.conclusions.copy()
            }
        
        try:
            response = self.agent.invoke({"input": query})
            
            output = response.get('output', '')
            
            if not output and 'intermediate_steps' in response:
                steps = response['intermediate_steps']
                if steps:
                    last_observation = steps[-1][1] if len(steps[-1]) > 1 else ""
                    output = f"Análise parcial baseada nas ferramentas executadas:\n\n{last_observation}"
            
            result = {
                'success': True,
                'response': output if output else "Não foi possível gerar uma resposta completa.",
                'plots': self.generated_plots.copy(),
                'conclusions': self.conclusions.copy()
            }
            
            self.generated_plots.clear()
            
            return result
            
        except Exception as e:
            plots = self.generated_plots.copy()
            self.generated_plots.clear()
            
            error_msg = str(e)
            
            if "iteration limit" in error_msg.lower() or "time limit" in error_msg.lower():
                return {
                    'success': True,
                    'response': "Executei algumas análises mas não consegui completar totalmente. Aqui estão os resultados parciais:",
                    'plots': plots,
                    'conclusions': self.conclusions.copy()
                }
            
            return {
                'success': False,
                'error': error_msg,
                'plots': plots,
                'conclusions': []
            }
    
    def _try_simple_answer(self, query: str) -> Optional[str]:
        """Tenta responder perguntas simples sem usar o agente"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['colunas', 'columns', 'variáveis', 'variables']):
            if 'quantas' in query_lower or 'número' in query_lower or 'how many' in query_lower:
                return f"O dataset possui {len(self.df.columns)} colunas: {', '.join(self.df.columns)}"
        
        if any(word in query_lower for word in ['linhas', 'rows', 'registros', 'records']):
            if 'quantas' in query_lower or 'quantos' in query_lower or 'how many' in query_lower:
                return f"O dataset possui {len(self.df)} linhas (registros)."
        
        if 'shape' in query_lower or 'tamanho' in query_lower or 'dimensão' in query_lower:
            return f"O dataset tem {self.df.shape[0]} linhas e {self.df.shape[1]} colunas."
        
        if 'primeiras' in query_lower or 'head' in query_lower:
            return f"Primeiras linhas do dataset:\n\n{self.df.head().to_string()}"
        
        return None


def main():
    """Interface principal do Streamlit"""
    
    st.title("🤖 Agente Autônomo de EDA")
    st.markdown("### Análise Exploratória de Dados com Inteligência Artificial")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configurações")

        if not OPENAI_API_KEY:
            st.error("❌ A variável `OPENAI_API_KEY` não foi encontrada no arquivo .env.")
        else:
            st.success("✅ API Key carregada do .env")

        st.markdown(f"**Modelo:** `{OPENAI_MODEL}`")

        st.markdown("---")
        
        st.header("📁 Upload do CSV")
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type=['csv'],
            help="Faça upload do arquivo CSV para análise"
        )
        
        st.markdown("---")
        st.header("ℹ️ Sobre")
        st.markdown("""
        Este agente autônomo pode:
        - ✅ Analisar dados
        - 📊 Criar visualizações
        - 🔍 Detectar anomalias
        - 🔗 Encontrar correlações
        - 📝 Gerar conclusões
        """)

    if not OPENAI_API_KEY:
        st.stop()
    
    if not uploaded_file:
        st.info("📁 Por favor, faça upload de um arquivo CSV na barra lateral para começar")
        return
    
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
        
        with st.expander("👀 Preview dos Dados", expanded=False):
            st.dataframe(df.head(10))
            st.write(f"**Shape:** {df.shape[0]} linhas × {df.shape[1]} colunas")
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
        return
    
    if 'agent' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
        with st.spinner("🤖 Inicializando agente..."):
            st.session_state.agent = EDAAgent(df, OPENAI_API_KEY, OPENAI_MODEL)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
    
    agent = st.session_state.agent
    
    st.header("💬 Chat com o Agente")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "plots" in message and message["plots"]:
                for plot in message["plots"]:
                    st.image(f"data:image/png;base64,{plot['image']}")
    
    if prompt := st.chat_input("Faça uma pergunta sobre os dados..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analisando..."):
                result = agent.analyze(prompt)
                
                if result['success']:
                    st.markdown(result['response'])
                    
                    if result['plots']:
                        for plot in result['plots']:
                            st.image(f"data:image/png;base64,{plot['image']}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['response'],
                        "plots": result['plots']
                    })
                else:
                    error_msg = f"❌ Erro: {result['error']}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    if st.sidebar.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()