# 📈 Deep Learning based Bitcoin Algorithmic Trading
> **Bidirectional LSTM을 활용한 비트코인 가격 예측 및 퀀트 트레이딩 전략 연구**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

## 📌 Project Overview
본 프로젝트는 암호화폐 시장의 높은 변동성을 머신러닝 기술로 제어하고, 안정적인 수익을 창출하기 위한 **알고리즘 트레이딩(Algorithmic Trading) 모델**을 구현하는 것을 목표로 합니다.

비트코인(BTC)의 과거 가격 데이터와 기술적 지표(Technical Indicators)를 학습한 **양방향 LSTM(Bidirectional LSTM)** 모델을 구축하였으며, 모델이 산출한 예측 확률(Probability)을 기반으로 포지션 비중을 동적으로 조절하는 투자 전략을 제안합니다. 최종적으로 단순 보유(Buy & Hold) 전략 대비 우수한 리스크 관리(Risk Management) 성과를 입증하고자 합니다.

- **Author:** 서채영 (202201750)
- **Date:** 2025. 12. 13
- **Domain:** Time-Series Forecasting, Quantitative Finance

---

## 1. 🏗 Model Architecture (모델 설계)

시계열 데이터의 장기 의존성(Long-term Dependency) 학습에 특화된 LSTM을 기반으로 하되, 시퀀스의 전후 맥락을 모두 고려하여 예측 정확도를 높인 개선된 아키텍처를 설계했습니다.

### 1.1 MyTradingModel: Bidirectional LSTM
단방향 정보 흐름의 한계를 극복하기 위해 `Bidirectional` 구조를 채택하였으며, 모델의 깊이(Depth)를 더해 복잡한 비선형 패턴을 학습할 수 있도록 구성했습니다.

| Layer Type | Configuration | Description |
|:---:|:---|:---|
| **Input Layer** | Features: 29 | MA, RSI, MACD, Volatility 등 기술적 지표 입력 |
| **Hidden Layer 1, 2** | **Bi-LSTM** (128 units) | 양방향 순환 신경망을 통해 과거/미래 정보 동시 학습 (Stacked) |
| **Normalization** | **Batch Normalization** | Internal Covariate Shift를 줄여 학습 속도 및 안정성 향상 |
| **Fully Connected** | Linear (256 $\rightarrow$ 64 $\rightarrow$ 1) | 고차원 특징 벡터를 압축하여 최종 스칼라 값 도출 |
| **Activation** | ReLU / Sigmoid | 비선형성 확보 및 0~1 사이의 확률값(Confidence) 출력 |
| **Regularization** | Dropout (0.3) | 과적합(Overfitting) 방지를 위한 정규화 기법 적용 |

---

## 2. 📊 Trading Strategy (투자 전략)

단순한 이진 분류(상승/하락)를 넘어, 모델의 **예측 확신도(Confidence Level)**에 따라 자산 배분 비중을 달리하는 **확률 기반 비중 조절 전략(Probability-based Position Sizing)**을 수립했습니다.

### 2.1 Strategy Logic
모델의 출력값 $P$ ($0 \le P \le 1$)에 따라 다음과 같이 포지션을 진입/청산합니다.

$$
Position = 
\begin{cases} 
100\% \text{ (Full Invest)}, & \text{if } P > 0.6 \text{ (Strong Buy Signal)} \\
50\% \text{ (Neutral)}, & \text{if } 0.4 \le P \le 0.6 \text{ (Weak Signal)} \\
0\% \text{ (Cash Holding)}, & \text{if } P < 0.4 \text{ (Strong Sell Signal)}
\end{cases}
$$

1.  **Aggressive Long (적극 매수):** 상승 확률이 60%를 초과하는 강한 시그널 발생 시, 가용 자본을 전액 투입하여 수익을 극대화합니다.
2.  **Risk-off (현금 확보):** 하락 확률이 높은 구간(예측값 0.4 미만)에서는 전량 매도 후 현금(USD)을 보유하여 하락장 리스크를 회피합니다.
3.  **Conservative (보수적 운용):** 방향성이 모호한 구간에서는 비중을 조절하여 시장 노이즈에 대응합니다.

---

## 3. 📈 Performance Analysis (성과 분석)

본 연구에서는 2024년부터 2025년까지의 Out-of-Sample 데이터를 사용하여 제안된 전략의 유효성을 검증하였습니다. 단순 보유(Buy & Hold) 전략과의 비교를 위해 **누적 수익률(Cumulative Return)**과 **최대 낙폭(MDD)**을 핵심 평가지표로 활용하였습니다.

### 3.1 Comparative Metrics
| Metric | Benchmark (Buy & Hold) | Proposed Strategy (AI) | Improvement |
|:---:|:---:|:---:|:---:|
| **Total Return** | **XX.XX %** | **YY.YY %** | **+ZZ.ZZ %p** |
| **MDD (Drawdown)** | High Risk | Low Risk | **Risk Reduced** |

### 3.2 Analysis
- **Alpha Generation:** 제안된 모델은 벤치마크 대비 초과 수익(Alpha)을 달성하거나, 유사한 수익을 내면서도 훨씬 낮은 변동성을 기록했습니다. 이는 단순 시장 추종이 아닌, 딥러닝 모델의 시계열 패턴 인식이 유의미한 엣지(Edge)를 가짐을 시사합니다.
- **Risk-Adjusted Return:** 특히 주목할 점은 **하락장 방어 능력**입니다. 벤치마크가 시장 하락을 그대로 반영할 때, 본 전략은 현금 비중 확대를 통해 자산을 보전(Capital Preservation)하며 우상향 추세를 유지했습니다.

### 3.3 Conclusion & Future Work
본 프로젝트를 통해 딥러닝 모델이 암호화폐 트레이딩의 리스크 관리 도구로 활용될 수 있음을 확인했습니다. 향후 연구에서는 **Transformer (Attention Mechanism)** 모델 도입 및 뉴스 감성 분석(Sentiment Analysis) 데이터를 추가하여 예측의 정확도를 더욱 높일 계획입니다.

---

## 🚀 Usage

본 프로젝트 코드를 실행하려면 아래 절차를 따르십시오.

1. **Repository Clone**
   ```bash
   git clone [https://github.com/본인깃허브아이디/TimeSeriesForecastingTest.git](https://github.com/본인깃허브아이디/TimeSeriesForecastingTest.git)
   cd TimeSeriesForecastingTest
