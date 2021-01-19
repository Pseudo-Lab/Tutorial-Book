#!/usr/bin/env python
# coding: utf-8

# # 3. 데이터 전처리

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yBd5UM049dC_XHex7Zx8D9ZO8mXfCElP?usp=sharing)

# In[5]:


from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/mgkUDA-V9oA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')




# 2장에서는 모델 학습에 사용할 데이터를 탐색하여 데이터 특성을 확인해보았습니다. 3장에서는 시계열 데이터 전처리 방법을 확인해보겠습니다.  
# 
# 시계열 데이터를 지도학습 문제로 변환 하기 위해서는 예측 대상이 되는 타겟 변수와 예측할 때 사용하는 입력 변수 쌍으로 데이터를 가공해야 합니다. 또한 딥러닝 모델을 안정적으로 학습시키기 위해선 데이터의 스케일(scale)을 통일 시키는 작업이 필요합니다. 3.1절에서는 코로나 확진자 데이터를 지도학습용 데이터로 변환하는 과정을 알아볼 예정이며, 3.2절에서는 데이터 스케일링 방법을 확인하겠습니다.  

# ## 3.1 지도학습용 데이터 구축

# 데이터 전처리 실습을 위해 앞서 2.1절에 나온 코드를 활용해 데이터를 불러오겠습니다.

# In[ ]:


get_ipython().system('git clone https://github.com/Pseudo-Lab/Tutorial-Book-Utils')
get_ipython().system('python Tutorial-Book-Utils/PL_data_loader.py --data COVIDTimeSeries')
get_ipython().system('unzip -q COVIDTimeSeries.zip')


# 2.3절에 나온 코드를 활용해 대한민국 일자별 확진자 데이터인 `daily_cases`를 산출하겠습니다.

# In[ ]:


import pandas as pd
confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
confirmed[confirmed['Country/Region']=='Korea, South']
korea = confirmed[confirmed['Country/Region']=='Korea, South'].iloc[:,4:].T
korea.index = pd.to_datetime(korea.index)
daily_cases = korea.diff().fillna(korea.iloc[0]).astype('int')
daily_cases


# 위와 같은 시계열 데이터를 모델이 지도학습에 사용할 수 있도록 입력 변수와 타겟 변수의 쌍으로 데이터를 가공해야 합니다. 시계열 문제에서는 이러한 데이터를 시퀀스(sequence) 데이터라고도 합니다. 시퀀스 데이터로 가공하기 위해서는 먼저 시퀀스 길이(sequence length)를 정의해야 합니다. 시퀀스 길이는 과거 몇 개의 데이터를 바탕으로 미래를 예측할지를 정합니다. 예를 들어 시퀀스 길이가 5인 경우 $t$ 시점을 예측하기 위해 과거 $t-1$, $t-2$, $t-3$, $t-4$, $t-5$ 시점의 데이터를 활용하게 됩니다. 이처럼 $t-k$ 부터 $t-1$ 까지의 데이터로 $t$ 시점을 예측하는 것을 one-step prediction 과제라고 칭합니다. 
# 
# 아래에 정의한 `create_sequences` 함수는 그림 3-1 처럼 크기가 N인 시계열 데이터를 N - seq_length 개의 지도학습용 데이터로 변환합니다.
# 
# ![](https://github.com/Pseudo-Lab/Tutorial-Book/blob/sungjin/pics/TS-ch3img01.png?raw=true)
# 
# 
# 
# 

# - 그림 3-1 시계열 데이터 변환 과정

# In[ ]:


import numpy as np

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)]
        y = data.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5
X, y = create_sequences(daily_cases, seq_length)


# `seq_length`를 5로 정의하고 `create_sequences` 함수를 `daily_cases`에 적용하니 총 327개의 지도학습용 데이터가 구축된 것을 확인할 수 있습니다. 

# In[ ]:


X.shape, y.shape


# 구축된 데이터를 학습용, 검증용, 시험용 데이터로 분리하겠습니다. 8:1:1 비율로 데이터를 분리하겠습니다. 327개의 80%는 약 261이므로 처음 261개 데이터를 학습용으로, 그 후 33개의 데이터를 검증용으로, 그리고 마지막 33개의 데이터를 시험용으로 사용하겠습니다.

# In[ ]:


train_size = int(327 * 0.8)
print(train_size)


# In[ ]:


X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+33], y[train_size:train_size+33]
X_test, y_test = X[train_size+33:], y[train_size+33:]


# In[ ]:


print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)


# ## 3.2 데이터 스케일링

# 이번 절에서는 데이터 스케일링을 실시하겠습니다. 데이터의 범위를 0과 1사이로 변환 시키는 MinMax scaling을 실시하겠습니다. MinMax scaling은 데이터 집합의 최소값과 최대값을 구한 뒤 아래 수식을 활용해 적용합니다. 
# 

# >$x_{scaled} = \displaystyle\frac{x - x_{min}}{x_{max} - x_{min}}$

# 스케일링 시 주의할 점은 훈련용 데이터의 통계량을 활용해 훈련용, 검증용, 시험용 데이터셋을 스케일링 해야 합니다. 시험용 데이터의 정보는 모델 학습시 입력되면 안되므로 훈련용 데이터의 통계량을 활용해 훈련용 데이터를 스케일링 합니다. 훈련용 데이터의 통계값으로 스케일링 된 데이터로 모델을 학습했기 때문에 추후 모델 성능 평가를 위해 입력되는 시험용 데이터에도 훈련용 데이터의 통계량으로 스케일링 합니다. 마찬가지로 검증용 데이터도 시험용 데이터가 겪게 되는 전처리 과정을 똑같이 적용해줘야 하기 때문에 훈련용 데이터의 통계량으로 스케일링 합니다.  
# 
# MinMax scaling을 적용하기 위해 `X_train` 데이터의 최소값과 최대값을 구하겠습니다.

# In[ ]:


MIN = X_train.min()
MAX = X_train.max()
print(MIN, MAX)


# 최소값이 0이고 최대값은 851입니다. 다음으로는 MinMax scaling 함수를 정의하도록 하겠습니다. 

# In[ ]:


def MinMaxScale(array, min, max):

    return (array - min) / (max - min)


# `MinMaxScale`함수를 활용해 스케일링을 진행하겠습니다

# In[ ]:


X_train = MinMaxScale(X_train, MIN, MAX)
y_train = MinMaxScale(y_train, MIN, MAX)
X_val = MinMaxScale(X_val, MIN, MAX)
y_val = MinMaxScale(y_val, MIN, MAX)
X_test = MinMaxScale(X_test, MIN, MAX)
y_test = MinMaxScale(y_test, MIN, MAX)


# 다음으로는 PyTorch 모델에 입력되기 위해 `np.array` 데이터 타입를 `torch.Tensor` 타입으로 변환해주겠습니다. 먼저 데이터 타입 변환하는 함수를 정의하겠습니다. 

# In[ ]:


import torch

def make_Tensor(array):
    return torch.from_numpy(array).float()


# `make_Tensor`함수를 활용해 데이터 타입 변환을 진행하겠습니다

# In[ ]:


X_train = make_Tensor(X_train)
y_train = make_Tensor(y_train)
X_val = make_Tensor(X_val)
y_val = make_Tensor(y_val)
X_test = make_Tensor(X_test)
y_test = make_Tensor(y_test)


# 지금까지 시계열 데이터를 지도학습용 데이터로 변환하는 방법과 데이터 스케일링 하는 방법을 확인해봤습니다. 다음 장에서는 구축된 데이터를 활용해 코로나 확진자 예측 모델을 구축해보겠습니다. 
