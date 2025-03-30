import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd





def plot_barras(df, color):
    # Configurar o estilo do gráfico
    sns.set(style="dark")

    for col in df.columns:
        if df[col].dtype == "object":
            agrupado = df[col].value_counts()

            # Configurar a paleta de cores
            colors = sns.color_palette(str(color), len(agrupado))

            # Criar o gráfico
            fig, ax = plt.subplots()
            bars = ax.bar(agrupado.index, agrupado.values, color=colors)

            # Adicionar rótulos nas barras
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{}'.format(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 pontos de deslocamento
                            textcoords="offset points",
                            ha='center', va='bottom')

            # Personalizar os eixos e o título
            ax.set_ylabel(f'Frequência de {col}', fontsize=10, fontweight="bold")
            ax.set_xlabel(f'Distribuição de {col}', fontsize=10, fontweight="bold")
            ax.set_title(f'Distribuição de {col}', fontsize=15, fontweight='bold')
            plt.xticks(rotation=45, ha='right')

            # Remover as bordas desnecessárias e ajustar layout
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            plt.show()
            
    return fig



def plot_hist(df, color):
    sns.set(style="whitegrid")  # Configurar o estilo do gráfico

    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            # Configurar o tamanho da figura
            fig, ax = plt.subplots()

            # Criar o histograma com barras coloridas
            sns.color_palette("cividis", len(df[col]))
            sns.histplot(df[col], kde=True, bins="auto", color=color, ax=ax)

            # Personalizar os eixos e o título
            ax.set_xlabel(f'Distribuição de {col}', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'Frequência de {col}', fontsize=14, fontweight='bold')
            ax.set_title(f'Histograma de {col}', fontsize=16, fontweight='bold')

            # Ajustar o layout para uma melhor apresentação
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()

            # Exibir o gráfico
            plt.show()

    return fig


def plot_boxplot(df, color):
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]: # Separação das colunas correspondentes
                                          
            #Configurações dos gráficos e dos eixos
            fig, ax = plt.subplots()
            sns.set(style="darkgrid")
            sns.boxplot(df, x=col, orient="horizontal", ax=ax, color=color)
            ax.set_xlabel(f"Distribuição de {col}", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"Frequência de {col}", fontsize=12, fontweight="bold")
            ax.set_title(f"Boxplot de {col}", fontsize=17, fontweight="bold")
            sns.despine(left=True, bottom=True, right=True)
            plt.tight_layout()
            
            plt.show()
            
    return fig
        
        
        
        
            
        
