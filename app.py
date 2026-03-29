import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# CARREGAR DADOS
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("enem.csv")
    return df

df = load_data()

# -----------------------------
# FUNÇÕES DE AMOSTRAGEM
# -----------------------------
def calcular_tamanho_amostra(N, z=1.96, p=0.5, e=0.05):
    n0 = (z**2 * p * (1 - p)) / (e**2)
    n = n0 / (1 + ((n0 - 1) / N))
    return int(n)

def amostra_aleatoria(df, n):
    return df.sample(n=n, random_state=42)

def amostra_sistematica(df, n):
    k = len(df) // n
    indices = np.arange(0, len(df), k)
    return df.iloc[indices[:n]]

def amostra_estratificada(df, coluna_estrato, n):
    return df.groupby(coluna_estrato, group_keys=False).apply(
        lambda x: x.sample(int(len(x) / len(df) * n), random_state=42)
    )

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
    "Correlação",
    "Amostragem"
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

    freq = df[variavel].value_counts().reset_index()
    freq.columns = [variavel, "Frequência"]
    freq["%"] = (freq["Frequência"] / freq["Frequência"].sum()) * 100

    st.subheader("Tabela de Frequência")
    st.write(freq)

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

    st.subheader("Estatísticas")
    st.write(f"Média: {df[variavel].mean():.2f}")
    st.write(f"Mediana: {df[variavel].median():.2f}")
    st.write(f"Desvio padrão: {df[variavel].std():.2f}")

    st.subheader("Histograma")
    fig, ax = plt.subplots()
    sns.histplot(df[variavel].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

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

# -----------------------------
# AMOSTRAGEM
# -----------------------------
elif opcao == "Amostragem":
    st.title("📌 Técnicas de Amostragem")

    colunas_num = [
        "nota_cn_ciencias_da_natureza",
        "nota_lc_linguagens_e_codigos",
        "nota_mt_matematica",
        "nota_redacao"
    ]

    variavel = st.selectbox("Escolha a variável:", colunas_num)

    metodo = st.selectbox("Tipo de amostragem:", [
        "Aleatória Simples",
        "Sistemática",
        "Estratificada"
    ])

    tipo_tamanho = st.radio("Tamanho da amostra:", [
        "20% dos dados",
        "Cálculo (95% confiança)"
    ])

    N = len(df)

    if tipo_tamanho == "20% dos dados":
        n = int(0.2 * N)
    else:
        n = calcular_tamanho_amostra(N)

    st.write(f"Tamanho da população: {N}")
    st.write(f"Tamanho da amostra: {n}")

    if metodo == "Aleatória Simples":
        amostra = amostra_aleatoria(df, n)

    elif metodo == "Sistemática":
        amostra = amostra_sistematica(df, n)

    elif metodo == "Estratificada":
        estrato = st.selectbox("Escolha o estrato:", [
            "sg_uf_prova",
            "regiao_nome_prova"
        ])
        amostra = amostra_estratificada(df, estrato, n)

    # -----------------------------
    # COMPARAÇÃO
    # -----------------------------
    st.subheader("📊 Comparação: População vs Amostra")

    media_pop = df[variavel].mean()
    media_amostra = amostra[variavel].mean()

    std_pop = df[variavel].std()
    std_amostra = amostra[variavel].std()

    comparacao = pd.DataFrame({
        "Métrica": ["Média", "Desvio Padrão"],
        "População": [media_pop, std_pop],
        "Amostra": [media_amostra, std_amostra]
    })

    st.write(comparacao)

    # -----------------------------
    # GRÁFICO
    # -----------------------------
    st.subheader("Distribuição")

    fig, ax = plt.subplots()
    sns.kdeplot(df[variavel].dropna(), label="População", ax=ax)
    sns.kdeplot(amostra[variavel].dropna(), label="Amostra", ax=ax)
    ax.legend()

    st.pyplot(fig)