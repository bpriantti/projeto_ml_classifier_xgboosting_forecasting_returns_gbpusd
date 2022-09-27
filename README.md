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
 - [Target Var Calc](#target-var-calc)
 - [Feature Eng](#feature-eng)
 - [Analise Correl Features](#analise-correl-features)
 - [Train Test Split](#train-test-split)

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

> Nesta Etapa realizou-se o cálculo da variável alvo, esta que será passada como base para o aprendizado supervisionado do modelo xg boosting, neste projeto a variável target é o retorno futuro em 2 dias em Pips, em seguida analisou-se a distribuição da variável alvo.

```
# Construcao dos alvos
periodos = 2

# target %
database["target"] = database["close"].pct_change(periodos).shift(-periodos)

# target em pips
database["target_pips"] = ((database["close"] - database["close"].shift(periodos))*10000).shift(-periodos)
```

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-02.png?raw=true"  width="800" height = "400">
 
 > Analisando a distribuição da variável alvo optou-se por realizar a classificação da mesma em:
 
 -  1: alta em 2 dias.
 -  0: variacoes entre -50 a 50 pips, em 2 dias.
 - -1: baixa em 2 dias.
 
 para isso utilizou-se o código a seguir:
 
 ```
# criacao do alvo binario:
database["target_bin"] = np.where(database['target_pips'] >  50 , 1, 0)
database["target_bin"] = np.where(database['target_pips'] < -50 ,-1, database['target_bin'])
```
 
 ### Feature Eng:
 
> A próxima etapa para o projeto é o processo de feature engineering que consiste em calcular as features para o modelo, estas que também são chamadas de variáveis dependentes, para isso utilizou-se o código a seguir:

```
#---:
def calc_features(df1):

    df1['var_1'] = ta.ADX(df1['high'],df1['low'],df1['close'],14)
    df1['var_2'] = ta.PLUS_DI(df1['high'],df1['low'],df1['close'],14)
    df1['var_3'] = ta.MINUS_DI(df1['high'],df1['low'],df1['close'],14)

    df1.dropna(inplace = True)
    return df1

dados = calc_features(database.copy())
```
> Obs: Para este projeto devido a particularidade do indicador adx, di + e di- variam em um range de 0 a 100 para a toda a base de dados, não é necessário o uso de filtros, normatizações, min-max scaler e discretização das features.

### Analise Correl Features:

> Em seguida analisou-se a correlação entre as features para um modelo com uma resposta estável torna-se interessante que as features sejam descorrelacionadas entre si, abaixo segue a tabela com os dados da correlação entre as features:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-05.PNG?raw=true"  width="380" height = "160">
   
### Train Test Split:

> kjj

```
#x-y split:
y = dados.loc[:,'target_pips':'target_bin']
X = dados.loc[:,'var_1':]

# Vamos treinar o modelo de 2009 a 2013
start_train = '1995'
end_train   = '2010'

# Vamos testar o modelo de 2014 a 2019
start_test = '2011'

y_train = y[start_train:end_train]
y_test  = y[start_test:]

x_train = X[start_train:end_train]
x_test  = X[start_test:] 
```
