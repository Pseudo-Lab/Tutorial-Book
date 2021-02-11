# 2. 데이터 탐색

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pseudo-Lab/Tutorial-Book/blob/master/book/chapters/time-series/Ch2-EDA.ipynb)

from IPython.display import YouTubeVideo

YouTubeVideo(id="9E130rYSzaE", width=560, height=315)



이전 장에서 시계열 분석에 활용 가능한 딥러닝 모델 구조와 모델 평가지표에 대해 알아보았습니다. 본격적으로 시계열 데이터를 사용하여 분석을 해보기 앞서 2장에서는 분석에 사용될 데이터셋에 대한 탐색과 시각화를 해보겠습니다.

시계열 예측에 사용할 데이터는 코로나 확진자 데이터입니다. 데이터셋은 [COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)와 [Kaggle: Novel Corona Virus 2019 Dataset
](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset?select=covid_19_data.csv)에서 확보했습니다. 

Johns Hopkins 대학의 저장소에는 매일 국가별 확진자 현황이 업데이트 되고 있으며, 이를 일자별 레포트 형태로 제공하고 있습니다. 일자별 레포트 양식을 변형해서 가공한 전체 기간의 레포트도 저장소에서 제공합니다. 또한 일자별 레포트 양식을 보존한 상태로 전체 데이터를 합한 데이터를 Kaggle에서 제공하고 있습니다. 

2.1절에서 해당 데이터를 다운받고 어떤 변수들로 이루어져 있는지 알아보겠습니다. 그리고 2.2절에서 전세계적으로 유행하고 있는 코로나인만큼 전세계 확진자로부터 추세를 알아보고 2.3절에서 대한민국 확진자를 더 자세하게 탐색해보도록 하겠습니다.

## 2.1 데이터 다운로드


먼저 코로나 확진자 데이터셋을 내려받도록 하겠습니다. 가짜연구소에서 제공하는 데이터 로더 함수를 사용하여 쉽게 받아볼 수 있습니다.

`git clone` 명령어를 사용하여 Tutorial-Book-Utils 저장소를 Colab 환경에 다운로드 합니다. \

!git clone https://github.com/Pseudo-Lab/Tutorial-Book-Utils

![dataset example](https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch2img01.PNG?raw=true)

- 그림 2-1 Tutorial-Book-Utils 저장소 폴더 경로

그림 2-1과 같이 저장소 내의 파일을 모두 다운받고 그 중에 `PL_data_loader.py` 파일이 위치한 것을 알 수 있습니다. 해당 파일에는 구글 드라이브의 데이터셋을 다운로드 하는 함수가 저장되어 있습니다. `--data` 파라미터에 `COVIDTimeSeries`을 입력하면 모델 구축에 활용할 코로나 확진자 수 데이터를 받을 수 있습니다.



!python Tutorial-Book-Utils/PL_data_loader.py --data COVIDTimeSeries

다음으로 리눅스 명령어인 `unzip`을 활용하여 압축파일을 풀어보도록 하겠습니다. `-q` 옵션을 통해 불필요한 출력물이 나오지 않게 제어 가능합니다.

!unzip -q COVIDTimeSeries.zip

![dataset example](https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch2img02.PNG?raw=true)

- 그림 2-2 다운로드된 데이터셋 파일

압축까지 풀면 그림 2-2와 같이 `covid_19_data.csv` 파일과 `time_series_covid19_confirmed_global.csv` 파일이 다운로드 된 것을 확인할 수 있습니다.

 `covid_19_data.csv` 파일을 활용해 시간에 따른 전세계 확진자 현황을 시각화 해볼 것이며, `time_series_covid19_confirmed_global.csv` 파일에서 대한민국 데이터를 활용해 확진자 수 변화를 시각화해보겠습니다. 우선 각 데이터를 읽어서 저장된 값을 확인해보겠습니다.

import pandas as pd

all = pd.read_csv('covid_19_data.csv')
confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')

`all`는 전세계의 일별 확진자, 사망자, 완치자에 대한 데이터입니다. `ObservationDate`는 발생 날짜, `Province/State`, `Country/Region`는 발생 지역과 국가, `Confirmed`는 확진자수, `Deaths`는 사망자수,	그리고 `Recovered`는 완치자수를 의미합니다. 해당 데이터프레임은 다음과 같습니다.

all

`confirmed`는 국가별 확진자 수에 대한 시퀀스(sequence) 데이터입니다. `Country/Region`, `Province/State`는 발생 지역과 국가, `Long`,	`Lat`는 경도와 위도, 그리고 `MM/DD/YYYY`는 일자별 확진자 수를 의미합니다. 해당 데이터프레임은 다음과 같습니다.

confirmed

## 2.2 전세계 데이터 EDA

`all`을 활용해 전세계 코로나 확진자 수를 시각화 해보도록 하겠습니다. 이 데이터프레임에는 위치를 나타내는 변수가 `Province/State`(지역)와 `Country/Region`(국가) 두 가지로 존재합니다. 그 중 국가별 확진자 수를 시각화 하기 위해서 `Country/Region` 기준으로 해당 국가의 모든 확진자 수를 더해줘야 합니다. 아래 코드를 통해 구현해보겠습니다. 





group = all.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum()
group = group.reset_index()
group.head()

일자별로 국가별 총 확진자 수가 도출 된 것을 확인할 수 있습니다. 다음으로는 세계지도 위에 국가별 확진자 수가 나타나지도록 애니메이션 효과로 표현해보겠습니다. 이는 `plotly.express` 패키지를 이용하여 나타낼 수 있습니다. 

import plotly as py
import plotly.express as px

먼저 `px.choropleth`로 지도 레이어를 만들고 `.update_layout`를 통해 날짜를 업데이트 하는 방식입니다.

각 파라미터의 의미는 다음과 같습니다.

- location : dataframe에서 위치를 나타내는 column

- locationmode : 나타낼 국가(‘ISO-3’, ‘USA-states’, ‘country names’중 하나)의 범위

- color : dataframe에서 그래프로 나타낼 column

- animation_frame : dataframe에서 애니메이션 효과를 줄 기준 column

재생버튼을 누르면 일별 확진자 그래프를 확인할 수 있습니다.

choro_map=px.choropleth(group, 
                    locations="Country/Region", 
                    locationmode = "country names",
                    color="Confirmed", 
                    animation_frame="ObservationDate"
                   )

choro_map.update_layout(
    title_text = 'Global Spread of Coronavirus',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
choro_map.show()

처음에는 중국에서 시작되어서 점점 전세계적으로 퍼지는 것을 볼 수 있습니다. 최근에는 북미, 남미, 그리고 인도의 확진자 수가 다른 지역에 비해 많은 것을 확인해볼 수 있습니다. 

## 2.3 대한민국 데이터 EDA



이번에는 `confirmed`에서 대한민국 데이터만 뽑아보겠습니다. 이 데이터는 누적데이터로 해당 날짜까지의 총 확진자수를 의미합니다.

confirmed[confirmed['Country/Region']=='Korea, South']

여기서 지역명, 위경도를 제외한 확진자수에 대한 정보만 남기도록 하겠습니다. 그리고 이후 편의성을 위해 행과 열을 바꾸고(`.T`), 인덱스인 날짜를 str 형식에서 datetime 형식으로 바꿔주었습니다(`to_datetime`).

korea = confirmed[confirmed['Country/Region']=='Korea, South'].iloc[:,4:].T
korea.index = pd.to_datetime(korea.index)
korea

가장 대표적인 시각화 패키지인 `matplotlib.pyplot`과 `seaborn`을 이용하여 시각화해보도록 하겠습니다. 이 때, `%matplotlib inline`은 해당 셀에서 바로 그래프를 보이게 할 수 있습니다. 그리고 `pylab`의 `rcParams['figure.figsize']`은 그래프 크기를 조정할 수 있으며, `sns.set`은 격자의 색깔과 글짜 크기를 조정할 수 있습니다.

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

plt.plot(korea)
plt.show()

이번에는 누적데이터가 아닌 일일 단위로 확진자 수를 알아보겠습니다. `diff`를 이용하면 이전 행과 차이를 구할 수 있어 누적데이터를 쉽게 일별데이터로 바꿀 수 있습니다. 하지만 첫번째 행에는 결측치가 생겨 이를 누적데이터의 첫번째 값으로 채워주어야 합니다. 그리고 데이터 형식은 int(정수)로 바꿔줍니다.

daily_cases = korea.diff().fillna(korea.iloc[0]).astype('int')
daily_cases

마찬가지로 일별 확진자수 데이터를 그래프로 나타내면 다음과 같습니다. 위에서 그래프 형식에 대한 셋팅을 미리 해두면 이후에는 똑같이 적용됩니다.

plt.plot(daily_cases)
plt.show()

3월초와 8월말에 급증하는 시기가 있고, 지속적으로 확진자가 발생하고 있는 것을 볼 수 있습니다. 그리고 연말이 되면서 다시 증가 추세를 보이고 있습니다.

이렇게 이번장에서 전세계와 대한민국의 코로나 확진자 수 데이터에 대해 알아보았습니다. 다음장부터는 대한민국 일별 코로나 확진자 수에 대해서 전처리하고 모델링하는 과정에 대해 알아보도록 하겠습니다.