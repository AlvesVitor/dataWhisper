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
    """Agente Autônomo para Análise Exploratória de Dados - Otimizado"""
    
    def __init__(self, df: pd.DataFrame, api_key: str, model_name: str):
        self.df = self._optimize_dataframe(df)
        self.df_original_size = len(df)
        self.conclusions = []
        self.analysis_history = []
        self.generated_plots = []
        
        # Configurações de otimização
        self.MAX_PLOT_POINTS = 10000  # Máximo de pontos em gráficos
        self.MAX_CORRELATION_COLS = 20  # Máximo de colunas para correlação
        
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

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Otimiza o dataframe reduzindo uso de memória"""
        df_optimized = df.copy()
        
        # Otimizar tipos numéricos
        for col in df_optimized.select_dtypes(include=['int']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        for col in df_optimized.select_dtypes(include=['float']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Converter object para category quando apropriado
        for col in df_optimized.select_dtypes(include=['object']).columns:
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:  # Se menos de 50% são únicos
                df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized

    def _get_sample_for_plot(self, df: pd.DataFrame, max_points: int = None) -> pd.DataFrame:
        """Retorna amostra do dataframe para plotagem"""
        if max_points is None:
            max_points = self.MAX_PLOT_POINTS
            
        if len(df) <= max_points:
            return df
        
        # Amostragem estratificada se possível
        return df.sample(n=max_points, random_state=42)

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

REGRAS IMPORTANTES:
- Use NO MÁXIMO 2 ações por pergunta
- Seja EXTREMAMENTE direto e objetivo
- Assim que tiver informação suficiente, vá IMEDIATAMENTE para "Final Answer"
- NUNCA repita ações já executadas
- Para gráficos: execute APENAS a ação de criar o gráfico e vá direto para Final Answer
- Para perguntas simples: use 1 ação e finalize

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
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor

    
    def _get_dataframe_info(self, _: str = "") -> str:
        """Retorna informações básicas do dataframe"""
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        
        info = f"""
Informações do Dataset:
- Shape: {self.df_original_size} linhas × {self.df.shape[1]} colunas
- Colunas: {list(self.df.columns)}
- Tipos de dados: {self.df.dtypes.to_dict()}
- Memória usada: {memory_usage:.2f} MB (otimizado)
"""
        return info

    def _get_statistical_summary(self, _: str = "") -> str:
        """Retorna resumo estatístico - otimizado"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return "Não há colunas numéricas no dataset."
        
        if len(numeric_cols) > 15:
            return f"Dataset com {len(numeric_cols)} colunas numéricas. Resumo das primeiras 15:\n{self.df[numeric_cols[:15]].describe().to_string()}"
        
        return self.df[numeric_cols].describe().to_string()

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
        """Calcula matriz de correlação - otimizado"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return "Não há colunas numéricas suficientes para calcular correlações."
        
        if len(numeric_cols) > self.MAX_CORRELATION_COLS:
            cols_to_use = numeric_cols[:self.MAX_CORRELATION_COLS]
            corr_matrix = self.df[cols_to_use].corr()
            return f"Matriz de Correlação (primeiras {self.MAX_CORRELATION_COLS} colunas):\n{corr_matrix.to_string()}"
        
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
        """Cria histograma - otimizado"""
        column = column.strip()
        
        if column not in self.df.columns:
            return f"Coluna '{column}' não encontrada."
        
        df_plot = self._get_sample_for_plot(self.df[[column]].dropna())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot[column].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title(f'Histograma: {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequência')
        
        if len(df_plot) < len(self.df):
            ax.text(0.02, 0.98, f'Amostra de {len(df_plot):,} pontos', 
                   transform=ax.transAxes, fontsize=9, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')  # Reduzir DPI
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
        """Cria gráfico de dispersão - otimizado"""
        try:
            col_x, col_y = [c.strip() for c in columns.split(',')]
        except:
            return "Formato inválido. Use: 'coluna_x,coluna_y'"
        
        if col_x not in self.df.columns or col_y not in self.df.columns:
            return f"Uma ou ambas as colunas não foram encontradas."
        
        df_plot = self._get_sample_for_plot(self.df[[col_x, col_y]].dropna())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_plot[col_x], df_plot[col_y], alpha=0.5, s=20)  # Pontos menores
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f'Scatter Plot: {col_x} vs {col_y}')
        
        if len(df_plot) < len(self.df):
            ax.text(0.02, 0.98, f'Amostra de {len(df_plot):,} pontos', 
                   transform=ax.transAxes, fontsize=9, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')  # Reduzir DPI
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
        """Cria box plot - otimizado"""
        column = column.strip()
        
        if column not in self.df.columns:
            return f"Coluna '{column}' não encontrada."
        
        df_plot = self._get_sample_for_plot(self.df[[column]].dropna())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot.boxplot(column=[column], ax=ax)
        ax.set_title(f'Box Plot: {column}')
        ax.set_ylabel(column)
        
        if len(df_plot) < len(self.df):
            ax.text(0.02, 0.98, f'Baseado em amostra de {len(df_plot):,} pontos', 
                   transform=ax.transAxes, fontsize=9, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')  # Reduzir DPI
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
                return f"O dataset possui {self.df_original_size:,} linhas (registros)."
        
        if 'shape' in query_lower or 'tamanho' in query_lower or 'dimensão' in query_lower:
            return f"O dataset tem {self.df_original_size:,} linhas e {self.df.shape[1]} colunas."
        
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
        
        **🚀 Otimizado para grandes datasets!**
        - Amostragem inteligente em gráficos
        - Redução automática de memória
        - Performance melhorada
        """)

    if not OPENAI_API_KEY:
        st.stop()
    
    if not uploaded_file:
        st.info("📁 Por favor, faça upload de um arquivo CSV na barra lateral para começar")
        return
    
    try:
        with st.spinner("📂 Carregando e otimizando dados..."):
            df = pd.read_csv(uploaded_file)
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        st.success(f"✅ Arquivo carregado: {uploaded_file.name}")
        
        with st.expander("👀 Preview dos Dados", expanded=False):
            st.dataframe(df.head(10))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Linhas", f"{df.shape[0]:,}")
            with col2:
                st.metric("Colunas", f"{df.shape[1]:,}")
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
        return
    
    if 'agent' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
        with st.spinner("🤖 Inicializando agente otimizado..."):
            st.session_state.agent = EDAAgent(df, OPENAI_API_KEY, OPENAI_MODEL)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            
            optimized_memory = st.session_state.agent.df.memory_usage(deep=True).sum() / 1024**2
            memory_saved = ((original_memory - optimized_memory) / original_memory) * 100
            
            if memory_saved > 5:
                st.info(f"💾 Memória otimizada: {original_memory:.1f}MB → {optimized_memory:.1f}MB (economia de {memory_saved:.1f}%)")
    
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