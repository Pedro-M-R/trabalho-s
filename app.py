import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CARREGAR DADOS
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("enem.csv")
    return df

df = load_data()

# -----------------------------
# FILTRO
# -----------------------------
st.sidebar.title("Filtros")

estado = st.sidebar.selectbox(
    "Filtrar por estado:",
    ["Todos"] + sorted(df["sg_uf_prova"].dropna().unique())
)

if estado != "Todos":
    df = df[df["sg_uf_prova"] == estado]

# -----------------------------
# MENU
# -----------------------------
st.sidebar.title("Menu")
opcao = st.sidebar.radio("Escolha a análise:", [
    "Visão Geral",
    "Variáveis Qualitativas",
    "Variáveis Quantitativas",
    "Correlação"
])

# -----------------------------
# VISÃO GERAL
# -----------------------------
if opcao == "Visão Geral":
    st.title("📊 Dashboard ENEM 2024")

    st.subheader("Dados")
    st.dataframe(df.head())

    st.subheader("Resumo Estatístico")
    st.write(df.describe())


# -----------------------------
# VARIÁVEIS QUALITATIVAS
# -----------------------------
elif opcao == "Variáveis Qualitativas":
    st.title("Análise de Variáveis Qualitativas")

    colunas_cat = [
        "sg_uf_prova",
        "nome_uf_prova",
        "regiao_nome_prova",
        "tp_status_redacao"
    ]

    variavel = st.selectbox("Escolha a variável:", colunas_cat)

    # Frequência
    freq = df[variavel].value_counts().reset_index()
    freq.columns = [variavel, "Frequência"]
    freq["%"] = (freq["Frequência"] / freq["Frequência"].sum()) * 100

    st.subheader("Tabela de Frequência")
    st.write(freq)

    # Gráfico
    st.subheader("Gráfico de Barras")
    fig, ax = plt.subplots()
    ax.bar(freq[variavel].astype(str), freq["Frequência"])
    plt.xticks(rotation=45)
    st.pyplot(fig)


# -----------------------------
# VARIÁVEIS QUANTITATIVAS
# -----------------------------
elif opcao == "Variáveis Quantitativas":
    st.title("Análise de Variáveis Quantitativas")

    colunas_num = [
        "nota_cn_ciencias_da_natureza",
        "nota_lc_linguagens_e_codigos",
        "nota_mt_matematica",
        "nota_redacao"
    ]

    variavel = st.selectbox("Escolha a variável:", colunas_num)

    # Estatísticas
    st.subheader("Estatísticas")
    st.write(f"Média: {df[variavel].mean():.2f}")
    st.write(f"Mediana: {df[variavel].median():.2f}")
    st.write(f"Desvio padrão: {df[variavel].std():.2f}")

    # Histograma
    st.subheader("Histograma")
    fig, ax = plt.subplots()
    sns.histplot(df[variavel].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot
    st.subheader("Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[variavel], ax=ax)
    st.pyplot(fig)


# -----------------------------
# CORRELAÇÃO
# -----------------------------
elif opcao == "Correlação":
    st.title("Matriz de Correlação")

    colunas_num = [
        "nota_cn_ciencias_da_natureza",
        "nota_lc_linguagens_e_codigos",
        "nota_mt_matematica",
        "nota_redacao"
    ]

    corr = df[colunas_num].corr()

    st.subheader("Tabela de Correlação")
    st.write(corr)

    st.subheader("Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)