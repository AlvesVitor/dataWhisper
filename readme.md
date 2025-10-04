# 🤖 Agente Autônomo de EDA - I2A2

> **Atividade Obrigatória - Desafio Extra**  
> Curso I2A2 - Inteligência Artificial Aplicada

### 🎯 Objetivo da Atividade

Criar um ou mais agentes que permitam a um usuário realizar perguntas sobre qualquer arquivo CSV disponibilizado, transformando a ferramenta em uma solução de EDA realmente útil e genérica.

## 📁 Dataset de Referência

O projeto foi desenvolvido e testado com o dataset de **Fraudes de Cartão de Crédito** disponível no Kaggle:

🔗 **Link**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Estrutura do Dataset

| Coluna       | Descrição                                            |
| ------------ | ---------------------------------------------------- |
| `Time`       | Número de segundos desde a primeira transação        |
| `V1` a `V28` | Features transformadas por PCA (privacidade)         |
| `Amount`     | Valor da transação                                   |
| `Class`      | Indicador de fraude: `1` = fraudulenta, `0` = normal |

> **Nota**: As colunas V1 a V28 foram transformadas através do algoritmo PCA para proteção de privacidade, portanto não é possível identificar seu conteúdo original.

## ✨ Funcionalidades

### 🔧 Ferramentas do Agente

- **📊 Análise Estatística Completa**: Estatísticas descritivas, correlações e distribuições
- **🔍 Detecção de Anomalias**: Identificação de outliers usando método IQR
- **📈 Visualizações Automáticas**:
  - Histogramas
  - Scatter plots
  - Box plots
- **🧹 Análise de Qualidade**: Verificação de valores faltantes e tipos de dados
- **💬 Interface Conversacional**: Interaja com seus dados através de linguagem natural
- **🤖 Agente Autônomo ReAct**: Decide automaticamente quais ferramentas usar
- **📝 Conclusões Inteligentes**: Gera e armazena insights sobre os dados

### 🎨 Interface Streamlit

- Upload de arquivos CSV
- Chat interativo com o agente
- Visualização de gráficos inline
- Histórico de conversas
- Preview dos dados

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit**: Interface web interativa
- **LangChain**: Framework para agentes LLM
- **OpenAI GPT-4**: Modelo de linguagem base
- **Pandas**: Manipulação de dados
- **Matplotlib/Seaborn**: Visualizações
- **NumPy**: Computação numérica

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd DataWhisper
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sua-chave-api-aqui
OPENAI_MODEL=gpt-4
```

> ⚠️ **Importante**: Nunca commite o arquivo `.env`! Adicione-o ao `.gitignore`

## 🚀 Como Usar

### Iniciando a Aplicação

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`

### Fluxo de Uso

1. **📤 Upload**: Faça upload de um arquivo CSV na barra lateral
2. **👀 Preview**: Visualize as primeiras linhas dos dados
3. **💬 Converse**: Faça perguntas em linguagem natural
4. **📊 Visualize**: Veja gráficos e análises gerados automaticamente

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────┐
│      Interface Streamlit            │
│  - Upload CSV                       │
│  - Chat Interface                   │
│  - Visualização de Gráficos         │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         EDAAgent Class              │
│  ┌────────────────────────────┐    │
│  │  LangChain ReAct Agent     │    │
│  │  - Reasoning               │    │
│  │  - Action Planning         │    │
│  │  - Tool Selection          │    │
│  └────────────┬───────────────┘    │
│               │                     │
│  ┌────────────▼───────────────┐    │
│  │    Memory & Context        │    │
│  │  - Conversation Buffer     │    │
│  │  - Analysis History        │    │
│  │  - Generated Plots         │    │
│  └────────────────────────────┘    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         9 Ferramentas               │
├─────────────────────────────────────┤
│  1. get_dataframe_info              │
│  2. get_statistical_summary         │
│  3. check_missing_values            │
│  4. get_correlation_matrix          │
│  5. detect_outliers                 │
│  6. create_histogram                │
│  7. create_scatter_plot             │
│  8. create_box_plot                 │
│  9. add_conclusion                  │
└─────────────────────────────────────┘
```

## 📝 Requisitos (requirements.txt)

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
langchain>=0.1.0
langchain-openai>=0.0.2
python-dotenv>=1.0.0
openai>=1.0.0
```
