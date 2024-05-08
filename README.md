# News_summ_finetuning
This repository is for summarization fine-tuning project

# Introduction

Huggingface 오픈소스를 활용한 Summarization Fine-tuning 프로젝트입니다. 이 프로젝트의 주요 학습 목표는 다음과 같습니다.


* Hugging Face Transformer 오픈소스 라이브러리 모델 활용
* Trainer API를 활용한 파인튜닝 스크립트 작성
* 실행 스크립트 작성
* 평가 지표 분석
* Hugging face 파인 튜닝 모델 배포
* requirement.txt, Docker 등 패키지 관리

## 1. 데이터 수집
AIhub의 생성요약(Abstractive Summarization)을 위한 한국어 데이터 세트를 활용했습니다.

[요약문 및 레포트 생성 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=582)

### 1.1. parameter 분석
- ID : Validation Set과 Train Set의 Genre 비율 대조

| Genre      | val/train    |    | Genre      | val/train    |
|------------|--------|----|------------|--------|
| public     | 1.00   |    | literature | 1.00   |
| edit       | 1.00   |    | narration  | 0.96   |
| speech     | 1.00   |    | his_cul    | 1.00   |
| minute     | 1.00   |    | news_r     | 1.00   |
| paper      | 1.00   |    | briefing   | 1.00   |

validation set와 train set의 비율 분포가 비슷한 클래스 분포를 가지고 있기 때문에 모델 학습과 훈련에 적합한 데이터 셋이다.



- CATEGORY

| | Count |
|--------|-------|
| REPORT | 146771 |

- PASSAGE

  
|         | Max Length | Min Length | Average Length   |
|----------------------|------------|------------|------------------|
| passage_train        | 1499       | 300        | 833.90           |
| passage_validation   | 1497       | 300        | 835.43           |

- SUMMARIES :'summaries'는 short_summary와 long_summary로 나뉜다.


| Metrics           | Max      | Min     | Avg               |
| ------------------- | -------- | ------- | ----------------- |
| train_summ_short_info | 100 | 15 | 76.01 |
| train_summ_long_info | 848 | 31 | 171.91 |
| val_summ_short_info | 100 | 20 | 75.00 |
| val_summ_long_info | 1095 | 40 | 173.41 |

| Metrics           | Max      | Min     | Avg               |
| ----------------- | -------- | ------- | ----------------- |
| train/val_short_info | 1.0 | 0.75 | 1.01 |
| train/val_long_info | 0.77 | 0.77625 | 0.99 |

### 1.2. 데이터 전처리

features: ['input_ids', 'attention_mask', 'labels']
- input_ids
  passage의 토큰화 결과
- attention_mask
  padding된 token을 나타내줌
  MAX_INPUT_LENGTH = 512
- labels
  summarization(short)의 토큰화 결과
  MAX_TARGET_LENGTH = 128
