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
 - [Treinando Avaliando XGboosting](#treinando-avaliando-xgboosting)
 - [Verificando Feature Importance](#verificando-feature-importance)
 - [Backtest Modelo](#backtest-modelo)
 - [Research Tuning XGboost Parâmetros](#research-tuning-xgboost-parâmetros)
 - [Re-Fit XGboosting Otimizado](#re-fit-xgboosting-otimizado)
 - [Feature Importance XGboosting Opt](#feature-importance-xgboosting-opt)
 - [Comparando Modelo Xgboosting No Opt vs With Opt](#comparando-modelo-xgboosting-no-opt-vs-with-opt)
 - [Conclusão e Trabalhos Futuros](#conclusão-e-trabalhos-futuros)

 
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

> Para realizar o treino e em seguida o teste do modelo, realizou-se a divisão da base em x_train,x_test,y_train e y_test abaixo segue o código utilizado:

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

> Podemos visualizar a separação dos dados com o seguinte com a imagem abaixo:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-06.PNG?raw=true" width="800" height = "400">

### Treinando-Avaliando XGboosting:

```
#import XGBoost:
import xgboost as xgb
from xgboost import XGBClassifier

#parameters:
params = {
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100,
            'ramdom_state':42
         }
            
#instanciando xgboosting:
xgb_clf = XGBClassifier(**params)

#fit:
xgb_clf.fit(x_train, y_train['target_bin'])

#parameters:
print(xgb_clf)

#prediçoes para o treinamento e teste:
y_train['pred'] = xgb_clf.predict(x_train)
y_test['pred']  = xgb_clf.predict(x_test)

#---:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#---:
print("------------------------------------------------------")
print('classification_report: train')
print(classification_report(y_train['target_bin'], y_train['pred']))
print("------------------------------------------------------")
print('classification_report: test')
print(classification_report(y_test['target_bin'], y_test['pred']))
```

__Resultado:__

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-07.PNG?raw=true" width="400" height = "400">
   
> Observou-se os parâmetros do modelo e verificou-se que para houve uma diferença entre as métricas para o treinamento e para o teste, isto se deve a grande componente aleatória para ativos no mercado financeiro, no entanto observamos que para o teste o modelo tem um recall e um precision médio de 0.30 para a compra e venda o que totaliza um accuracy médio de 0.60 para o modelo, podendo assim ser útil observar o desempenho do modelo em um backtest e verificar o retorno da abordagem.

### Verificando Feature Importance:
   
> O XGboosting é uma excelente ferramenta também para análise bivariada das features com a variável alvo, verificou-se o feature importance para as features no entanto como são poucas não optou-se por retirar alguma feature e retreinar o modelo.
   
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-08.PNG?raw=true" width="600" height = "300">
   
### Backtest Modelo:
   
> Para verificar a performance do algoritmo como estratégia de investimento, realizou-se o backtest do modelo, verificou-se que o mesmo obteve um bom desempenho acumulando um total de 18 mil PIPS nos períodos de dados desconhecidos de 2011 a 2022 com um stop fixo de 90 pips.

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-09.PNG?raw=true" width="800" height = "400">

### Research Tuning XGboost Parâmetros:

> Com o objetivo de melhorar a performance do modelo realizou-se o processo de tuning dos hiperparâmetros pelo método empírico, inicialmente realizou-se um estudo do comportamento do modelo variando apenas um parâmetro com os outros mantidos constantes, os parâmetros utilizados para a otimização foram:

- Max Depth
- Alpha
- Learnig Rate
- N estimators

> Abaixo, tem-se o desempenho dos testes de comportamento para quando se varia cada parâmetro.

__Comportamento Opt Max Depth:__

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-10.PNG?raw=true" width="580" height = "300">

__Comportamento Opt Alpha:__

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-11.PNG?raw=true" width="580" height = "300">

__Comportamento Opt Learning Rate:__

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-12.PNG?raw=true" width="580" height = "300">
   
__Comportamento Opt N estimators:__

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-13.PNG?raw=true" width="580" height = "300">

__Rank Modelos Otimizados:__

> Com a análise do comportamento dos modelos observou-se que os parâmetros alpha e n estimators possuem pouco efeito sobre a performance do modelo, diante deste fato optou-se por otimizar apenas os parâmetros:

- Max Depth
- Learning Rate

> Após realizar o treinamento e teste do modelo variando os parâmetros acima, realizou-se um rank para verificar a melhor parametrização encontrada em termos de precision score, como demonstrado na tabela abaixo:

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-14.PNG?raw=true" width="280" height = "180">
 
### Re-Fit XGboosting Otimizado:

> Encontrado os melhores parâmetros para o modelo, realizou-se o processo de re-fit que consiste em re-treinar o modelo com as novas parametrizações encontradas na otimização a partir deste ponto verificou-se também as métricas de avaliação de um modelo de classificação observando a melhora em termos de precision, recall e também f1 score. 

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-15.PNG?raw=true" width="400" height = "400">
   
### Feature Importance XGboosting Opt:

> Verificou-se também o feature importance para o modelo e não se observou grandes mudanças em relação a este score.

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-16.PNG?raw=true" width="600" height = "300">
   
### Backtest Modelo Otimizado:

> Com o objetivo de realizar uma análise da performance do modelo realizou-se o processo de backtest, que consiste em testar uma estratégia no passado, observou-se que o modelo otimizado teve um acumulado de 24 mil pips.

<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-17.PNG?raw=true" width="800" height = "400">
   
### Comparando Modelo Xgboosting No Opt vs With Opt:

> Para verificar qual foi o melhor modelo, realizou-se a comparação entre o modelo otimizado e o não otimizado, observou-se a performance superior do modelo otimizado e também verificou-se que o mesmo não possui anos negativos, sendo esta uma performance satisfatória pois foi um total de 10 anos de teste em dados desconhecidos.

__Comparando Equity Backtest dos Modelos:__
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-18.PNG?raw=true" width="800" height = "400">

__Comparando Pips Acumulados dos Modelos:__
<p align="center">
   <img src="https://github.com/bpriantti/projeto_ml_classifier_xgboosting_forecasting_returns_gbpusd/blob/main/images/image-19.PNG?raw=true" width="800" height = "400">

### Conclusão e Trabalhos Futuros:

> Pode-se concluir que o uso do xgboosting para o predict de retornos futuros em 2 dias realiza uma performance satisfatória, resolvendo o problema listado no bussines problem, destaca-se que o modelo de xgboosting nao tem anos negativos sendo isto um feito admirável no desenvolvimento, como trabalhos futuros planeja-se melhorar o hypertunning dos parâmetros e incluir mais features de entropia para o modelo.
