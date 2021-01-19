# 코로나 확진자 수 예측 모델 구축 

코로나 확진자 수 예측 모델 구축 튜토리얼에 오신 것을 환영합니다. 

해당 교육용 튜토리얼은 **PyTorch를 활용한 LSTM 모델 구축 실습에 목적**이 있으며, 확진자 예측에 대한 **전문적인 견해를 제시하는 용도는 아님**을 밝힙니다. 

이번 튜토리얼의 개요는 다음과 같습니다. 

**사용 데이터셋**: 

1. JHU CSSE COVID-19 Data (출처: [Github](https://github.com/CSSEGISandData/COVID-19))
2. Novel Corona Virus 2019 Dataset (출처: [Kaggle](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset?select=covid_19_data.csv))

**실습 환경**: Google Colaboratory

**모델 구조**: LSTM, CNN-LSTM

**목차**: 

- Time Series 소개
- 데이터 탐색
- 데이터 전처리
- LSTM
- CNN-LSTM
- 참고 문헌

가짜연구소는 오픈(open)의 가치를 추구합니다. 피드백은 언제든지 환영하며 각 페이지별로 댓글을 달아주시면 되겠습니다. 