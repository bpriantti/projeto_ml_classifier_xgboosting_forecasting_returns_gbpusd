# Projeto Ml Classifier XGboosting Forecasting Returns - GBP-USD

__Bussines Problem:__

> Acertar o melhor momento para realizar uma operação de investimento não é uma tarefa fácil, muitas vezes tentar utilizar estratégias de investimento de forma discricionária pode causar prejuízos financeiros a investidores de grande e pequeno porte, de posse desta informação decidiu-se por meio de técnicas de machine learning desenvolver um modelo quantitativo para prever a direção do ativo GBP-USD(libra esterlina dólar) em comprar, vender e esperar, e por meio da tecnologia e data science conseguir melhor resultados de investimento.

__Objetivo:__

> Desenvolver um modelo de machine learning para prever o retorno futuro em 2 dias do ativo GBP-USD, classificando os retornos em alta, baixa e esperar, com base em critérios com base em um threshold definido com base na distribuição dos retornos. Neste projeto optou-se por utilizar o algoritmo de XGboosting.

__Autor:__  
   - Bruno Priantti.
    
__Contato:__  
  - bpriantti@gmail.com

__Encontre-me:__  
   -  https://www.linkedin.com/in/bpriantti/  
   -  https://github.com/bpriantti
   -  https://www.instagram.com/brunopriantti/
   
__Frameworks Utilizados:__

- Numpy: https://numpy.org/doc/  
- Pandas: https://pandas.pydata.org/
- Matplotlib: https://matplotlib.org/ 
- Seaborn: https://seaborn.pydata.org/  
- Plotly: https://plotly.com/  
- Scikit learn: https://scikit-learn.org/stable/index.html
- Statsmodels: https://www.statsmodels.org/stable/index.html
- TA-lib: https://mrjbq7.github.io/ta-lib/doc_index.html
___

## Contents
 - [Importando Libraries](#importando-libraries) 
 - [Data Request](#data-request) 
 - [Data Wralling](#data-wralling)
 - [Data Visualization](#data-visualization)
 - [Feature Calculation](#feature-calculation)
 - [Train Test Split](#train-test-split)
 - [Elbow Method Clusters](#elbow-method-clusters)
 - [Treinando Modelo K-Means](#treinando-modelo-k-means)
 - [Bivariate Analisys](#bivariate-analisys)
 - [Backtest Base de Teste](#backtest-base-de-teste)
 - [Backtest Base Completa](#backtest-base-completa)

### Importando Libraries:
> inicialmente para este projeto realizou-se o import das bibliotecas que serao utilizadas para machine learning, data wralling e data visualization dos dados, utilizou-se os comandos abaixo para esta etapa:

```
#import libs:
import pandas as pd
import numpy as np

#libs para visualization dos dados:
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md

#---:
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as Dash
from plotly.subplots import make_subplots

#lib para api com ativos da bolsa:
import talib as ta

#---
import warnings
warnings.filterwarnings("ignore")
```

### Data Request:

> O primeiro passo para o desenvolvimento é importar a base de dados do ativo GBP-USD, esta que já foi previamente tratada e encontra-se disponível em um repositório do github do próprio autor, abaixo tem-se os comandos para a execução do acesso à base de dados. 

```
url = 'https://raw.githubusercontent.com/bpriantti/projeto_ml_classifier_forecasting_returns_model_eval/main/files/GBPUSD_Daily_199305120000_202208030000.csv'

data = pd.read_csv(url,sep = '\t')
```

### Data Wralling:

> Em seguida realizou-se o processo de data wralling, que consiste em tratamentos na base de dados para posterior uso dos dados para o desenvolvimento do modelo de machine learning e backtesting, realizou-se esta etapa com o código abaixo:

```
#renanme colunas:
data.columns = ['date', 'open','high','low','close','tickvol','vol','spread']
data.drop(['tickvol','vol'],axis = 1, inplace = True)

#atualizando index para data:
data['Date'] = pd.to_datetime(data['date']).to_frame()
data.set_index(data['Date'],inplace = True)
data.drop(['Date','date'],axis =1, inplace = True)

#utilizando dados apenas de 95 para frente:
data = data['1995':]
```
### Data Visualization:

> Após o tratamento dos dados no processo de data wralling, realizou-se o processo de Data Visualization, que consiste em visualizar a base de dados, para o caso atual foi realizada esta etapa para verificar possíveis inconsistências visíveis na série histórica.

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-01.png?raw=true"  width="800" height = "460">
   

   
### Target Var Calc: 

> Nesta Etapa realizou-se o cálculo da variável alvo, esta que será passada como base para o aprendizado supervisionado do modelo xg boosting, neste projeto a variavel target é o retorno futuro em 2 dias em Pips.

```
# Construcao dos alvos
periodos = 2

# Alvo 1 - Retorno
database["target"] = database["close"].pct_change(periodos).shift(-periodos)

# Variaçao em Pips do alvo
database["target_pips"] = ((database["close"] - database["close"].shift(periodos))*10000).shift(-periodos)
```

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_clustering_de_indicadores_de_timming_para_estrategia_de_quant_trading./blob/main/images/image-2.PNG?raw=true"  width="800" height = "460">
 
 ### Train-Test Split:
 
 > Para realizar o treino e em seguida o teste do modelo, realizou-se a divisão da base em data-train e data-test.
 
```
#data train-test split:
data_train = database.loc['2005-02-14':'2012-12-31']
data_test = database.loc['2013-01-01':]

#features x, train-test:
x_train = data_train.loc[:,'adx':'neg_dir_mov']
x_test = data_test.loc[:,'adx':'neg_dir_mov']
```

### Elbow Method Clusters:

> Utilizou-se o método de elbow method para verificar o número de clusters necessário, verificou-se que clusters entre 4 e 6 demonstraram bons resultados de agrupamento:
   
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_clustering_de_indicadores_de_timming_para_estrategia_de_quant_trading./blob/main/images/image-3.png?raw=true"  width="800" height = "460">

### Treinando Modelo K-Means:

> Em seguida realizou-se o treinando do modelo com 5 clusters, e visualizou-se os clusters em um plot 3d e também o histograma de frequência para os clusters. 

```
#fit k-means model:
kmodel = KMeans(n_clusters = 5, random_state = 1)
clusters = kmodel.fit(k)
```

```
#data visualization:
df = data_train.loc[:,'adx':]

fig = px.scatter_3d(df, x='pos_dir_mov', y='neg_dir_mov', z='adx',
                    color='clusters',height=500, width=800, title = 'Clusters Plot: ')

fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))

fig.show()
```
- Clusters Plot:
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_clustering_de_indicadores_de_timming_para_estrategia_de_quant_trading./blob/main/images/cluster_3d.png?raw=true"  width="740" height = "460">

- Histograma Frequencia Clusters:
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_clustering_de_indicadores_de_timming_para_estrategia_de_quant_trading./blob/main/images/image-4.PNG?raw=true"  width="680" height = "300">

### Bivariate Analisys:

> Com o objetivo de verificar qual cluster teve a melhor performance como regra de negociação, realizou-se uma análise bivariada entre os clusters e a mediana dos retornos futuros em 2 dias, em seguida utilizou-se os comandos sort para rankear os clusters em mais e menos lucrativos, como demonstrado na imagem abaixo.

```
#acessando os clusters
data_train['clusters'] = clusters.labels_

#rank por mediana:
rank = data_train[['target_var','clusters']].groupby(['clusters']).median()
rank.sort_values(by = ['target_var'], ascending = False)*100

#plot:
cmap = sns.diverging_palette(10, 133, as_cmap=True)
g = sns.heatmap(data= rank.sort_values(by = ['target_var'], ascending = False), cmap='coolwarm_r',cbar=True,linewidths=.1,annot=True,fmt=".1%",annot_kws={'rotation':0},center=0.00,xticklabels=True)
g.figure.set_size_inches(w=27/2.54, h=18/2.54)
g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 10)
g.set_title('Rank Clusters Mediana Retornos Futuros')
plt.show();
```
   
- Rank Clusters Mediana Retornos Futuros:   
   
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_clustering_de_indicadores_de_timming_para_estrategia_de_quant_trading./blob/main/images/image-5.PNG?raw=true"  width="400" height = "320">

> Utilizou-se como regra de negociação, comprar a ação quando o clusters for 0,3 e vender quando o cluster for 4.
   
### Backtest Base de Teste:
   
> Verificando histograma de frequencia base de teste:
 
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_clustering_de_indicadores_de_timming_para_estrategia_de_quant_trading./blob/main/images/image-6.png?raw=true"  width="680" height = "400">
   
- Backtest em juros simples para a base de test:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_clustering_de_indicadores_de_timming_para_estrategia_de_quant_trading./blob/main/images/image-7.png?raw=true"  width="680" height = "400">
   
### Backtest Base Completa:

> Em seguida realizou-se o backtest para a base completa, realizando o sizing de 30% alocado por negociação de um capital de R$ 100.000,00, obteve-se um lucro líquido fictício aproximado de 270.000,00 para um drawdown máximo de 10%.  
   
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_clustering_de_indicadores_de_timming_para_estrategia_de_quant_trading./blob/main/images/image-8.png?raw=true"  width="680" height = "440">
   
 
 
