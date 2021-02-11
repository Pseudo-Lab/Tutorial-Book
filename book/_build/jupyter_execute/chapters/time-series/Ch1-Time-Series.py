# 1. Time Series 소개

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pseudo-Lab/Tutorial-Book/blob/master/book/chapters/time-series/Ch1-Time-Series.ipynb)

from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/DRZFhCBsGQY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


시계열 예측은 과거에 관측된 값을 바탕으로 미래 값을 예측하는 문제입니다. 과거에 관측된 데이터와 미래 값 사이의 패턴을 발견해야 한다는 점에서 지도학습 문제로 정의가 가능합니다. 그렇기 때문에 이번 장에서는 신경망 구조에 기반한 지도학습을 통해 미래 값을 예측하는 모델을 구축해보겠습니다. 

시계열 예측은 다방면에서 필요로 하는 기술입니다. 가장 대표적으로 에너지 분야가 있습니다. 전력발전소에서는 효율적인 예비전력 확보를 위해 미래의 전력 수요를 예측해야 하며, 도시가스 회사는 검침기 고장 및 검침기 치팅에 대한 선제적 조치를 하기 위해 미래의 도시가스 사용량 예측 모델이 필요합니다. 실제로 해당 문제들은 새로운 모델 발굴을 위해 데이터 경진대회([전력](https://dacon.io/competitions/official/235606/overview/), [도시가스](https://icim.nims.re.kr/platform/question/16))로도 개최가 됐습니다. 이 외에도 유통 분야에서도 효율적인 물품 관리를 위해 품목별 판매량 예측에 관심있으며, 마찬가지로 데이터 경진대회([유통](https://www.kaggle.com/c/m5-forecasting-accuracy/overview))로도 개최가 됐습니다.

이번 튜토리얼에서는 Johns Hopkins University의 Center for Systems Science and Engineering에서 제공하는 [코로나 확진자 데이터](https://github.com/CSSEGISandData/COVID-19)를 활용해 과거 확진자 데이터를 바탕으로 미래 확진자를 예측하는 모델을 구축해보겠습니다. 1장에서는 시계열 예측 모델 구축 시 사용 가능한 신경망 구조에 대해 알아 볼 것이며, 모델 성능 평가 시 사용 가능한 평가지표를 확인해보겠습니다. 2장에서는 데이터 탐색적 분석을 통해 코로나 확진자 데이터에 대한 이해를 심화시킬 것이며 3장에서는 시계열 데이터를 지도학습을 위한 데이터 형식으로 바꾸는 법을 알아볼 것입니다. 4장과 5장에서는 각각 딥러닝 모델을 활용해 미래 확진자를 예측해보겠습니다. 


## 1.1 사용 가능한 딥러닝 구조

### 1.1.1 CNN

<p align="center">
<img align="center" src="https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch1img01.PNG?raw=true">
</p>





- 그림 1-1 CNN 적용 예시 (출처: Lim et al. 2020. Time Series Forecasting With Deep Learning: A Survey)  


일반적으로 CNN은 컴퓨터 비전 문제에서 우수한 성능을 보이는 네트워크 구조입니다. 허나 시계열 예측에서도 CNN 적용이 가능합니다. 1차원 Convolution 필터를 활용해 입력되는 sequence 데이터간의 가중합(weighted sum)을 구하여 예측 대상인 미래 값을 산출할 수 있습니다. 허나 CNN 구조는 과거와 미래 데이터간에 시간적인 의존성에 대해서는 고려하지 않습니다. 

### 1.1.2 RNN

<p align="center">
<img align="center" src="https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch1img02.PNG?raw=true">
</p>






- 그림 1-2 RNN 적용 예시 (출처: Lim et al. 2020. Time Series Forecasting With Deep Learning: A Survey)  

RNN은 자연어 처리 문제에서 자주 활용되는 구조로써 이전 상태의 정보가 축적된 은닉 상태(hidden state) 정보를 활용해 미래 예측에 활용합니다. 그렇기 때문에 과거의 정보를 활용해 미래의 예측값을 산출 할 수 있습니다. 하지만 주어지는 입력 sequence가 너무 방대할 경우 모델 학습에 악영향을 미치는 vanishing gradient 문제가 발생할 수 있습니다. 그렇기 때문에 해당 문제를 해결한 LSTM 구조를 주로 활용하고 있으며, 이번 튜토리얼에서도 LSTM 구조를 사용할 예정입니다. 

### 1.1.3 Attention Mechanism

<p align="center">
<img align="center" src="https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch1img03.PNG?raw=true">
</p>


- 그림 1-3 Attention Mechanism 적용 예시 (출처: Lim et al. 2020. Time Series Forecasting With Deep Learning: A Survey)  

과거 정보별 미래 예측시 도움되는 정보와 도움이 되지 않는 정보가 있을 것입니다. 예를 들어 유통업자가 주말 매출을 예측하고자 할 시에는 하루 전날인 평일의 매출 보다는 일주일 전의 같은 날인 주말 매출을 고려하는 것이 도움 될 수 있습니다. Attention 메커니즘을 활용한다면 이러한 예측이 가능해집니다. 과거 시점 별 예측 하고자 하는 시점에 미치는 영향력을 산출해서 미래 값 예측 시 사용하게 됩니다. 예측하고자 하는 시점과 과거에 있는 값 중에 직접적으로 연관 있는 값에 더 많은 가중치를 부여함으로써 보다 정확한 예측이 가능합니다. 

## 1.2 평가지표

이번 튜토리얼에서는 코로나 확진자 예측 모델을 구축해볼 예정입니다. 확진자는 연속된 값을 지니고 있기 때문에 예측 된 값과 실제 값 사이의 차이 값을 통해 모델의 성능을 가늠할 수 있습니다. 이번 절에서는 예측 값과 실제 값 사이의 차이를 계산하는 다양한 방법을 알아보겠습니다. 평가지표를 설명하기에 앞서 여러 기호들에 대한 정의를 먼저 실시하겠습니다.  




> $y_i$: 예측 대상인 실제 값 \
$\hat{y}_i$: 모델에 의한 예측 값 \
$n$: 시험 데이터셋(test dataset)의 크기  




1.2.1절 부터 1.2.4절 까지는 위의 기호를 사용하며, 1.2.5절에서는 기호 정의가 달라지게 되므로 이 점에 주의 부탁드립니다.

### 1.2.1 MAE (Mean Absolute Error)

>$MAE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} |y_i-\hat{y}_i|$

L1 Loss로도 불리는 MAE는 예측한 값과 실제 값의 차이에 절대값을 취해 모두 더해주고나서 계산한 샘플 갯수(n)로 나눠서 구할 수 있습니다. 샘플 갯수 만큼 모두 더한 후 나눠준다는 것은 평균을 구한다는 것이므로, 앞으로 나오는 평가지표들에 대해서는 평균을 구한다는 표현을 사용하겠습니다. MAE의 스케일(scale)은 예측 대상인 타겟 변수와 같은 스케일이기 때문에 값이 내포하는 의미를 직관적으로 이해하기에 좋습니다. 코드로 구현하면 아래와 같습니다. 

import numpy as np #넘파이 패키지 불러오기

def MAE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.abs(true-pred))

TRUE = np.array([10, 20, 30, 40, 50])
PRED = np.array([30, 40, 50, 60, 70])

MAE(TRUE, PRED)

### 1.2.2 MSE (Mean Squared Error)


>$MSE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} (y_i-\hat{y}_i)^2$

>$RMSE=\sqrt{\frac{1}{n}\displaystyle\sum_{i=1}^{n} (y_i-\hat{y}_i)^2}$



L2 Loss로도 불리는 MSE는 예측값과 실제값의 차이를 제곱하여 산출한 후 평균을 구한 값입니다. 예측값이 실제값에서 더 많이 벗어날 수록 MSE 값은 기하 급수적으로 증가하는 특성을 지닙니다. 산출된 값은 제곱이 되었기 때문에 타겟 변수와 값의 스케일이 다릅니다. 타겟 변수와 스케일을 맞춰주기 위해서는 MSE값에 루트를 씌워줄 수 있으며 이 값을 RMSE라고 부릅니다. 코드로 구현하면 아래와 같습니다.

def MSE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.square(true-pred))

TRUE = np.array([10, 20, 30, 40, 50])
PRED = np.array([30, 40, 50, 60, 70])

MSE(TRUE, PRED)

### 1.2.3 MAPE (Mean Absolute Percentage Error)

>$MAPE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} |\frac{y_i-\hat{y}_i}{y_i}|$



(출처: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)

MAPE는 실제값과 예측값 사이의 차이를 실제값으로 나눠줌으로써 오차가 실제값에서 차지하는 상대적인 비율을 산출합니다. 그리고 해당 값을 절대값 취한 후 평균을 구합니다. 오차의 정도를 백분율 값으로 나타내기 때문에 모델의 성능을 직관적으로 이해하기 쉬우며, 타겟 변수가 여러개 일 때 각 변수별 모델의 성능을 평가하기 용이합니다.

하지만 실제값에 0이 존재한다면 MAPE가 정의 되지 않는 문제점이 있습니다. 또한 절대적인 값의 오차가 같더라도 실제값과 예측값과의 대소 관계에 따라 과대 추정하는 예측값에 패널티를 더 부여하는 문제가 있습니다([Makridakis, 1993](https://doi.org/10.1016/0169-2070(93)90079-3)). 이는 아래 코드를 통해 확인해보겠습니다. 

def MAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.abs((true-pred)/true))

TRUE_UNDER = np.array([10, 20, 30, 40, 50])
PRED_OVER = np.array([30, 40, 50, 60, 70])
TRUE_OVER = np.array([30, 40, 50, 60, 70])
PRED_UNDER = np.array([10, 20, 30, 40, 50])


print('평균 오차가 20일 때 실제값과 예측값의 대소 관계에 따른 MAE, MAPE 비교 \n')

print('실제값이 예측값 보다 작을 때 (예측값이 과대추정)')
print('MAE:', MAE(TRUE_UNDER, PRED_OVER))
print('MAPE:', MAPE(TRUE_UNDER, PRED_OVER))


print('\n실제값이 예측값 보다 클 때 (예측값이 과소추정)')
print('MAE:', MAE(TRUE_OVER, PRED_UNDER))
print('MAPE:', MAPE(TRUE_OVER, PRED_UNDER))


MAPE는 산식 특성 상 백분율로 변환하기 위해서 실제값인 $y$로 나눠주는 방법을 취하고 있습니다. 그러므로 도출되는 값이 $y$에 의존적인 특성을 지니고 있습니다. 분자가 같더라도 분모가 더 작아지면 오차가 커지게 됩니다. 

위의 코드에서는 실제값이 예측값보다 20 만큼 작은 (`TRUE_UNDER`, `PRED_OVER`)와 20 만큼 큰 (`TRUE_OVER`, `PRED_UNDER`)를 통해 이를 확인했습니다. MAE 값은 `TRUE_UNDER`와 `PRED_OVER`, 그리고 `TRUE_OVER`와 `PRED_UNDER` 모두 20으로 같습니다. 하지만 MAPE는 실제값이 `TRUE_UNDER`일 경우 0.913, `TRUE_OVER`일 경우 0.437를 산출하고 있습니다. 

### 1.2.4 SMAPE (Symmetric Mean Absolute Percentage Error)



>$SMAPE=\frac{100}{n}\displaystyle\sum_{i=1}^{n} \frac{|y_i-\hat{y}_i|}{|y_i| + |\hat{y}_i|}$  




(출처: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)  


SMAPE는 앞서 언급한 예시에 대해 MAPE가 지닌 한계점을 보완하기 위해 고완됐습니다([Makridakis, 1993](https://doi.org/10.1016/0169-2070(93)90079-3)). 아래 코드를 통해 확인해보겠습니다.

def SMAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) #100은 상수이므로 이번 코드에서는 제외

print('평균 오차가 20일 때 실제값과 예측값의 대소 관계에 따른 MAE, SMAPE 비교 \n')

print('실제값이 예측값 보다 작을 때 (예측값이 과대추정)')
print('MAE:', MAE(TRUE_UNDER, PRED_OVER))
print('SMAPE:', SMAPE(TRUE_UNDER, PRED_OVER))


print('\n실제값이 예측값 보다 클 때 (예측값이 과소추정)')
print('MAE:', MAE(TRUE_OVER, PRED_UNDER))
print('SMAPE:', SMAPE(TRUE_OVER, PRED_UNDER))


MAPE는 0.91, 0.43의 다른 값을 산출했지만 SMAPE는 0.29의 같은 값을 산출하는 것을 확인할 수 있습니다. 하지만 SMAPE는 분모에 예측값 $\hat{y}_i$이 들어가서 $\hat{y}_i$에 의존적인 특성을 지니고 있습니다. 예측값이 과소추정할 때 분모가 더 작아지므로 계산되는 오차가 커지는 현상이 발생합니다. 아래 코드를 통해 확인해보겠습니다. 

TRUE2 = np.array([40, 50, 60, 70, 80])
PRED2_UNDER = np.array([20, 30, 40, 50, 60])
PRED2_OVER = np.array([60, 70, 80, 90, 100])

print('평균 오차가 20일 때 과소추정, 과대추정에 따른 MAE, SMAPE 비교 \n')

print('과대추정 시')
print('MAE:', MAE(TRUE2, PRED2_OVER))
print('SMAPE:', SMAPE(TRUE2, PRED2_OVER))

print('\n과소추정 시')
print('MAE:', MAE(TRUE2, PRED2_UNDER))
print('SMAPE:', SMAPE(TRUE2, PRED2_UNDER))

`PRED2_UNDER`와 `PRED2_OVER`모두 `TRUE2`와 평균 20의 오차를 지니고 있지만, SMAPE는 과소추정한 `PRED2_UNDER`에 대해 0.218의 값을 계산하는 반면 과대추정한 `PRED2_OVER`에 대해서는 0.149의 값을 계산합니다.

### 1.2.5 RMSSE (Root Mean Squared Scaled Error)

>$RMSSE=\sqrt{\displaystyle\frac{\frac{1}{h}\sum_{i=n+1}^{n+h} (y_i-\hat{y}_i)^2}{\frac{1}{n-1}\sum_{i=2}^{n} (y_i-y_{i-1})^2}}$

RMSSE 산식의 기호 정의 부터 진행하겠습니다. 각각의 기호는 아래와 같은 의미를 지니고 있습니다.

> $y_i$: 예측 대상인 실제 값  
>
> $\hat{y}_i$: 모델에 의한 예측 값  
>
> $n$: 훈련 데이터셋(train dataset)의 크기
>
> $h$: 시험 데이터셋(test dataset)의 크기

RMSSE는 Mean Absolute Scaled Error([Hyndman, 2006](https://doi.org/10.1016/j.ijforecast.2006.03.001))의 변형된 형태로써 앞서 언급한 MAPE와 SMAPE가 지닌 문제점을 해결합니다. MAPE와 SMAPE는 MAE를 스케일링(scaling)하기 위해 시험 데이터의 실제값과 예측값을 활용하기 때문에 오차의 절대값이 같아도 과소, 과대추정 여부에 따라 패널티가 불균등하게 부여됩니다. 

RMSSE는 MSE를 스케일링 할 때 훈련 데이터를 활용하므로 이러한 문제에서 벗어납니다. 훈련 데이터에 대해 naive forecasting을 했을 때의 MSE 값으로 나눠주기 때문에 모델 예측값의 과소, 과대 추정에 따라 오차 값이 영향을 받지 않습니다. naive forecast 방법은 가장 최근 관측값으로 예측하는 방법으로, 아래와 같이 정의됩니다. 

> $\hat{y}_i = y_{i-1}$

$i$ 시점의 예측값을 $i-1$ 시점의 실제값으로 예측하는 방법입니다. naive forecast 방법에 대한 MSE값으로 나눠주기 때문에, RMSSE값이 1보다 크면 naive forecast 방법보다 예측을 못한다는 뜻이며 1보다 적으면 naive forecast 방법보다 예측을 잘한다는 의미를 지닙니다. 아래 코드를 통해 RMSSE를 구현해보겠습니다. 

def RMSSE(true, pred, train): 
    '''
    true: np.array 
    pred: np.array
    train: np.array
    '''
    
    n = len(train)

    numerator = np.mean(np.sum(np.square(true - pred)))
    
    denominator = 1/(n-1)*np.sum(np.square((train[1:] - train[:-1])))
    
    msse = numerator/denominator
    
    return msse ** 0.5

TRAIN = np.array([10, 20, 30, 40, 50]) #RMSSE 계산을 위한 임의의 훈련 데이터셋 생성

print(RMSSE(TRUE_UNDER, PRED_OVER, TRAIN))
print(RMSSE(TRUE_OVER, PRED_UNDER, TRAIN))
print(RMSSE(TRUE2, PRED2_OVER, TRAIN))
print(RMSSE(TRUE2, PRED2_UNDER, TRAIN))

오차의 절대값이 같지만 MAPE와 SMAPE가 불균등하게 패널티를 부여했던 4개의 예시에 대해 균등하게 패널티가 부여됬으며 스케일링도 이뤄진것을 확인할 수 있습니다. 

지금까지 시계열 예측시 사용가능한 딥러닝 구조와 평가지표에 대해 알아보았습니다. 다음 장에서는 모델 구축에 활용할 코로나 확진자 데이터셋을 탐색해보겠습니다. 