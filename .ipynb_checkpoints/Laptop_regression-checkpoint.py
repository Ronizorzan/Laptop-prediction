#Bibliotecas necessárias
import streamlit as st
import pandas as pd
from numpy.random import randint as randint
from numpy import reshape
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import shap



#Configuração do Layout
st.set_page_config(page_title="Aplicação para Previsão de Preços de Laptops", layout="wide")


#Função principal
@st.cache_resource
def load_and_process_data():
    data = pd.read_csv("laptop_prices.csv")
    desvio = data["Price"].std()
    data = data[(data["Price"]> 0.5 * desvio) & (data["Price"]<= 4* desvio)] # Filtragem de outliers

    
    #Separação  entre variáveis independentes e variável dependente
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    X_treino, X_teste, y_treino, y_teste = train_test_split(X,y, test_size = 0.3, random_state=3219)
   
    
    #Codificação de categorias e salvamanento do objeto em dicionário para uso posterior
    encoders = {}
    for col in X_treino.columns:
        encoder = LabelEncoder()
        if X_treino[col].dtype== "object":
            X_treino[col] = encoder.fit_transform(X_treino[col])
            X_teste[col] = encoder.transform(X_teste[col])
            encoders[col] = encoder


    #Seleção de Atributos
    seletor = SelectKBest(f_regression, k=7)
    seletor.fit(X_treino, y_treino)
    colunas_selecionadas = X.columns[seletor.get_support()]
    X_treino, X_teste = seletor.transform(X_treino), seletor.transform(X_teste) 

    #Criação do Modelo com os parâmetros encontrados pelo otimizador bayesiano
    modelo = XGBRegressor(objective="reg:squarederror", colsample_bytree= 0.5, learning_rate= 0.1, max_depth= 20, max_leaves= 30, n_estimators= 1000, subsample= 1.0)
    modelo.fit(X_treino, y_treino)

    # Previsão e criação de métricas para validação
    previsoes = modelo.predict(X_teste)

    rmse = root_mean_squared_error(y_teste, previsoes)

    mape = mean_absolute_percentage_error(y_teste, previsoes)             
              

    
    #Retorno de variáveis que vamos utilizar posteriormente
    return mape, rmse, data, colunas_selecionadas, modelo, encoders, X_teste




#Criação da barra lateral
with st.sidebar:
    st.header("Selecione as Configurações:")    
    mape, rmse, data, colunas_selecionadas, modelo, encoders, X_teste = load_and_process_data()
    
    #Inserção de novos dados para previsão
    st.markdown("*Calcule um resumo com Laptops de todas as marcas*")
    todas_as_marcas = st.checkbox("Calcular resumo para todas as marcas", value=False)
    st.markdown("*Selecione as características do laptop.*")    
    novos_dados = [st.selectbox("Selecione a Marca", data["Brand"].unique()),
                    st.selectbox("Selecione o Processador", data["Processor"].unique()),
                    st.selectbox("Insira a quantidade de Memória RAM (GB) ", data["RAM (GB)"].unique()),
                    st.selectbox("Selecione o Armazenamento", data["Storage"].unique()),
                    st.selectbox("Selecione a GPU", data["GPU"].unique()),
                    st.selectbox("Selecione o Tamanho da Tela", data["Screen Size"].unique()),
                    st.selectbox("Selecione a resolução", data["Resolution"].unique()),
                    #st.selectbox("Selecione o Sistema Operacional", data["Operating System"].unique())
                    ]
    processar = st.button("Processar os dados")
if processar:
    with st.spinner("Aguarde... Carregando os dados"):

        #Tabulações
        tab1, tab2 = st.tabs(["Modelo", "Gráficos de Explicabilidade"])
        
        
        with tab1:
            col1, col2 = st.columns([0.55,0.45], gap="large")
            with col1:

                #Transformação dos novos dados em dataframe e codificação de categorias                
                novos_dados = pd.DataFrame([novos_dados], columns=colunas_selecionadas) 
                for col in novos_dados.columns:
                        if col in encoders:
                            novos_dados[col] = encoders[col].transform(novos_dados[col])


                #Plotagem interpretação dos novos dados únicos
                if not todas_as_marcas:
                    fig, ax = plt.subplots()
                    sns.set_style(style="darkgrid")
                    
                    explainer = shap.Explainer(modelo, feature_names=colunas_selecionadas)
                    novos_dados_reshaped = reshape(novos_dados, (1,-1))
                    marca_dec = encoders["Brand"].inverse_transform(novos_dados.loc[:,"Brand"])
                    explain = explainer(novos_dados_reshaped)
                    shap_values = explain.values
                    colors = ["red" if values <=0 else "green" for values in shap_values[0]]
                    plt.bar(colunas_selecionadas, shap_values[0], color=colors, width=0.9)                 
                    ax.set_title(f"Interpretação dos valores para a Marca: {marca_dec[0]}", fontsize=16, fontweight="bold")
                    ax.set_xlabel("Componentes do Laptop", fontsize=12, fontweight="bold")
                    ax.set_ylabel("Impacto dos componentes na previsão", fontsize=12, fontweight="bold")
                    plt.grid(True, linestyle="solid", linewidth=0.4, color="grey")
                    plt.xticks(rotation=30, ha="right")
                    st.pyplot(fig, use_container_width=True)
                    st.markdown("<hr style='border:1px solid green'>", unsafe_allow_html=True)
                    st.markdown(f"*O gráfico acima exibe em detalhes os componentes que tiveram impacto substancial \
                                (em verde) no valor do laptop, já as barras vermelhas indicam os componentes que \
                                impulsionaram negativamente o valor dos Laptops da Marca:* **{marca_dec[0]}** ")
                    


                #Plotagem interpretação dos novos dados múltiplos
                else:
                    fig, ax = plt.subplots()
                    sns.set_style(style="darkgrid")

                    explainer = shap.Explainer(modelo, feature_names=colunas_selecionadas)
                    novos_dados_unique = reshape(novos_dados.iloc[0,:], (1,-1))
                    marca_decod = encoders["Brand"].inverse_transform(novos_dados.loc[:,"Brand"])
                    explain = explainer(novos_dados_unique)
                    shap_values = explain.values
                    unique_color = ["red" if values<= 0 else "green" for values in shap_values[0]]
                    plt.bar(colunas_selecionadas, shap_values[0], color=unique_color, width=0.9)
                    ax.set_title(f"Interpretação dos valores para a Marca: {marca_decod[0]}", fontsize=16, fontweight="bold")
                    ax.set_xlabel("Componentes do Laptop", fontsize=12, fontweight="bold")
                    ax.set_ylabel("Impacto dos componentes na previsão", fontsize=12, fontweight="bold")
                    plt.grid(True, linestyle="solid", linewidth=0.4, color="grey")
                    plt.xticks(rotation=30, ha="right")
                    st.pyplot(fig, clear_figure=True, use_container_width=True)      
                    st.markdown("<hr style='border:1px solid green'>", unsafe_allow_html=True)          
                    st.markdown(f"*O gráfico acima exibe em detalhes os componentes que tiveram impacto substancial \
                                (em verde) no valor do laptop, já as barras vermelhas indicam os componentes que \
                                impulsionaram negativamente o valor dos Laptops da Marca:* **{marca_decod[0]}**")


        
            #Coluna de exibição das métricas e explicações
            with col2: 
                marcas_unicas = encoders["Brand"].transform(data["Brand"].unique())

                #loop para previsão de múltiplas marcas
                if todas_as_marcas:
                    previsoes = []
                    for marca in marcas_unicas:
                        novos_dados.iloc[0,0] = marca
                        previsto = modelo.predict(novos_dados)
                        previsoes.append(float(previsto))
                    marcas_unicas = encoders["Brand"].inverse_transform(marcas_unicas)
                    previsoes_df = pd.DataFrame([previsoes], columns=marcas_unicas)

                    
                    #Exibição das Métricas e resumos das previsões múltiplas
                    st.markdown("<h1 style='color: grey;'>Resultados das previsões</h1>", unsafe_allow_html=True)                                
                    st.markdown("Descubra abaixo os valores previstos para diferentes marcas de laptops com a mesma configuração, \
                                além de um resumo com o valor médio, menor valor e maior valor.")                
                    st.markdown("**Valores gerados para todas as marcas (em $)**")
                    st.dataframe(previsoes_df)
                    st.markdown("*Valor Médio dos Laptops: em ($):* **{:,.2f}**".format(previsoes_df.values.mean()))
                    minimo = previsoes_df.values.min()
                    maximo = previsoes_df.values.max()
                    marca_minimo = previsoes_df.columns[previsoes_df.values[0]==minimo]
                    marca_maximo = previsoes_df.columns[previsoes_df.values[0]==maximo]
                    st.markdown("*Menor valor entre os Laptops em ($):*  **{:,.2f}** - Marca: **{}**".format(minimo, marca_minimo[0]))
                    st.markdown("*Maior Valor entre os Laptops em ($):*  **{:,.2f}** - Marca: **{}**".format(maximo, marca_maximo[0]))                
                    st.markdown("<hr style='border:1px solid green'> ", unsafe_allow_html=True)  
                
                else:
                    previsao = modelo.predict(novos_dados.values) #Exibição da previsão única
                    st.markdown("<h1 style='color: grey;'>Resultados da previsão</h1>", unsafe_allow_html=True)
                    st.markdown("O valor aproximado do laptop é: $**{:,.2f}**:".format(previsao[0]))
                    st.markdown("<hr style='border:1px solid green'> ", unsafe_allow_html=True)  

                
                #Exibição das Métricas do modelo

                st.markdown("*O gráfico de barras ao lado revela de forma detalhada como cada componente individual afeta o preço do laptop. \
                            As barras verdes representam impacto positivo no preço, enquanto as vermelhas impactam negativamente o preço do laptop.*")                            
                st.markdown(f"Mean Absolute Percentage Error (MAPE): **{mape*100:.2f}**%")                
                st.markdown(f"Root Mean Squared Error (RMSE): **{rmse:.2f}**")
                st.markdown(f"A análise de desempenho do modelo mostra um MAPE de {mape*100:.2f}% e um RMSE de {rmse:.2f}, \
                            indicando a alta precisão das previsões e a confiança que você pode ter nos resultados do modelo.")
                
                


        #Tabulação de explicabilidade do modelo
        with tab2:        
        
            col1, col2 = st.columns([0.45,0.55], gap="large")
            
            #Criação e exibição do gráfico de explicabilidade global
            with col1: 
                importancia = modelo.feature_importances_
                fig2, ax2= plt.subplots()
                sns.set_style(style="darkgrid")
                
                
                imp_colors = ["red" if imp <=0.15 else "green" for imp in importancia] #Lista para seleção de cores vermelha ou verde de acordo com o valor
                plt.barh(colunas_selecionadas, importancia, color=imp_colors, height=0.9)
                sns.despine(right=True, bottom=True)
                st.markdown("<h3 style='color: gray;'>Impacto dos Componentes: Decisões Gerais</h3>", unsafe_allow_html=True)
                ax2.set_xlabel("Impacto sobre as previsões", fontsize=12, fontweight="bold")
                ax2.set_ylabel("Componentes do Laptop", fontsize=12, fontweight="bold")
                ax2.set_title("Impacto dos Componentes sobre as previsões", fontsize=16, fontweight="bold")
                plt.yticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                st.markdown("*O gráfico acima mostra o impacto geral dos componentes sobre o valor dos laptops*")
                st.markdown("Este gráfico revela a influência global de cada componente nos valores dos laptops. Observe como a Memória RAM, Resolução e Processador impulsionam o valor do laptop, \
                             enquanto determinadas GPUs e Armazenamentos podem reduzir significativamente o preço!")

            
            #Explicabilidade Local
            with col2:

                    instancia_escolhida = randint(0, X_teste.shape[0]) #Seleção aleatória para explicabilidade local
                    row = X_teste[ instancia_escolhida,:] 
                    row_reshaped = reshape(row, (1,-1))
                
                    #Criação dos valores Shap e do gráfico para plotagem
                    explainer = shap.Explainer(modelo)
                    explain = explainer(row_reshaped)
                    
                    fig3, ax3 = plt.subplots()
                    sns.set_style(style="darkgrid")                               

                    shap_values = explain.values
                    shap_colors = ["red" if values<= 0 else "green" for values in shap_values[0]]
                    plt.barh(colunas_selecionadas, shap_values[0], color=shap_colors, height=0.9)
                    sns.despine(bottom=True, left=True, right=True)
                    ax3.set_title("Impacto dos componentes do laptop sobre uma previsão única", fontweight="bold", fontsize=16)
                    st.markdown("<h3 style='color: gray;'> Impacto dos componentes: Decisões Individuais</h>", unsafe_allow_html=True)
                    ax3.set_xlabel("Valores dos componentes", fontweight="bold", fontsize=12)
                    ax3.set_ylabel("Componentes do Laptop", fontweight="bold", fontsize=12)
                    plt.yticks(rotation=30, ha="right")
                    st.pyplot(fig3, use_container_width=True)
                    st.markdown("**O gráfico acima mostra a influência de cada componente sobre um laptop específico**")
                    st.markdown("**Aqui, visualizamos uma análise detalhada do impacto de cada componente em uma configuração específica de laptop.**")                    
                    st.markdown(f"*Valor Previsto pelo Modelo para essa configuração: $*  **{modelo.predict(row_reshaped)[0]:,.2f}**")

                    df = pd.DataFrame(row_reshaped, columns=colunas_selecionadas)
                    for coluna in str(df.columns):
                        if coluna in encoders:
                            df[coluna] = encoders[coluna].inverse_transform(df[coluna])

                    
                    st.markdown("**O Dataframe abaixo possui a configuração original do laptop mostrado no gráfico acima**")
                    st.write(df)                             

            