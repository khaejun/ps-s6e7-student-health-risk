# Playground Series S6E7 — Predicting Student Health Risk

Kaggle: https://www.kaggle.com/competitions/playground-series-s6e7

## 진행 상황
- [x] 대회/데이터 파악
- [x] 전처리 (`notebooks/01_preprocessing.ipynb`)
- [x] EDA (`notebooks/02_eda.ipynb`)
- [x] 라벨 생성 규칙 역추적 — 팀원 분석, 검증 완료
- [x] CV 체계 + 사전확률 보정 + 결측 복구 피처 (`notebooks/03_cv_and_prior_correction.ipynb`) — **OOF balanced accuracy 0.94980**
- [x] LightGBM 하이퍼파라미터 튜닝 (`notebooks/04_lgbm_tuning.ipynb`) — **0.94987 (+0.00007, 예상대로 개선폭 미미)**
- [x] `stress_level` 예측 대치 시도 (`notebooks/05_stress_level_imputation.ipynb`) — **실패(기각). -0.00025, 최종본은 04 그대로 유지**
- [x] `sleep_duration` 회귀 기반 정밀 대치 (`notebooks/06_sleep_duration_regression.ipynb`) — **애매한 결과. 대치 자체는 개선(RMSE ↓, 경계판정 ↑)됐지만 전체 CV는 -0.00013로 노이즈 수준. 최종본은 04 그대로 유지**
- [x] Interaction 피처 + missing_count + Native NaN 실험 (`notebooks/07_interaction_and_native_nan.ipynb`, 팀원 XGBoost 리포트 아이디어 차용) — **전부 실패. 최종본은 04 그대로 유지**
- [ ] 결측 2개 이상 겹친 행 처리 — 다음 단계 (남은 유일한 실질적 개선 여지, 다만 표본이 작아 기대 효과는 제한적)
- [ ] 제출

## 프로젝트 구조
```
가이드_프로젝트/
├── PROJECT_GUIDE.md
├── notebooks/
│   ├── 01_preprocessing.ipynb            # 전처리 (완료)
│   ├── 02_eda.ipynb                      # EDA (완료)
│   ├── 03_cv_and_prior_correction.ipynb  # CV+사전확률보정+결측복구 (완료)
│   ├── 04_lgbm_tuning.ipynb              # Optuna 하이퍼파라미터 튜닝 (완료)
│   ├── 05_stress_level_imputation.ipynb  # stress_level 예측 대치 실험 (완료, 기각)
│   ├── 06_sleep_duration_regression.ipynb # sleep_duration 회귀 대치 실험 (완료, 기각)
│   └── 07_interaction_and_native_nan.ipynb # interaction/missing_count/native NaN 실험 (완료, 기각)
└── playground-series-s6e7/
    ├── train.csv / test.csv / sample_submission.csv   # 원본
    └── processed/
        ├── train_processed.csv     # 01 전처리 결과 (690,088 x 34)
        ├── test_processed.csv      # 01 전처리 결과 (295,753 x 33)
        ├── cv_folds.csv            # 팀 공용 고정 CV 폴드 (id, fold)
        ├── submission_v1.csv       # 03 결과 (사전확률 보정, 튜닝 전)
        └── submission_v2_tuned.csv # 04 결과 (사전확률 보정 + Optuna 튜닝) — 현재 최종본
```

> **커널 안내**: `03`~`05` 노트북은 LightGBM(+Optuna)이 필요합니다. base(anaconda) 환경엔 없고 `teammate` conda 환경에 설치돼 있어서, 그 환경을 Jupyter 커널로 등록(`Python (teammate)`)해서 실행했습니다. 이 노트북들을 열 때는 커널을 **Python (teammate)**로 선택하세요.
>
> **제출 안내**: `sample_submission.csv`는 전 행이 `at-risk`로만 채워진 템플릿 파일입니다 (BA 0.3333). Kaggle에 제출할 때 이 파일이 아니라 `processed/submission_v2_tuned.csv`를 올려야 합니다 — 실제로 이 둘을 헷갈려서 0.3333이 나온 적이 있었음.

## 과제 개요
- **문제 유형**: 3-클래스 다중분류
- **타겟 변수**: `health_condition` — `at-risk` / `unhealthy` / `fit`
- **평가지표**: **Balanced Accuracy** (예측 클래스와 실제 클래스 간 balanced accuracy)
- **참가 규모**: 6,691명 참가 신청 / 3,452명 참가자 / 3,355팀 / 34,190회 제출
- **난이도 태그**: Beginner, Tabular

## 타임라인
- 대회 시작: 2026-07-01
- 최종 제출 마감: 2026-07-31 (Entry/Team Merger 마감도 동일)
- 마감 시각은 별도 명시 없으면 UTC 11:59 PM 기준
- **현재 상태: Late Submission** (마감 지남 — 순위 산정용 제출은 불가하지만 학습/연습용 제출은 가능한 상태로 보임)

## 시상
- 1~3위: Kaggle 굿즈(merchandise) 선택 지급, Points/Medal 없음
- 1인당 시리즈 내 1회만 수상 가능 (초보자 참여 유도 목적)

## 데이터 성격
- Tabular Playground Series 특성상 실제 데이터를 기반으로 합성 생성한 데이터셋 → 완전히 현실적이지는 않을 수 있음(아티팩트 가능성 존재)
- 가볍게(light-weight) 여러 모델·피처 엔지니어링 아이디어를 빠르게 반복해보라는 취지의 대회

## 데이터 규모
| 파일 | 행 수 | 열 수 | 비고 |
|---|---|---|---|
| train.csv | 690,088 | 15 | id, target 포함 |
| test.csv | 295,753 | 14 | target 없음 |
| sample_submission.csv | 295,753 | 2 | id, health_condition |

- id 중복 없음 (train/test 각각 unique)

## 타겟 분포 — 심한 불균형
| 클래스 | 개수 | 비율 |
|---|---|---|
| at-risk | 592,561 | 85.9% |
| unhealthy | 57,724 | 8.4% |
| fit | 39,803 | 5.8% |

> 평가지표가 balanced accuracy(클래스별 recall의 평균)이므로, 다수 클래스(at-risk)만 잘 맞혀서는 점수가 오르지 않음. 소수 클래스(fit, unhealthy) recall을 끌어올리는 것이 핵심 — class weight, 오버/언더샘플링, threshold 조정 등 전략 검토 필요.

## 피처 (13개)

**수치형 (8)**
- `sleep_duration` (수면 시간)
- `heart_rate` (심박수)
- `bmi`
- `calorie_expenditure` (칼로리 소모)
- `step_count` (걸음 수)
- `exercise_duration` (운동 시간)
- `water_intake` (수분 섭취량)

**범주형 (6)**
- `diet_type`: veg / non-veg / balanced
- `stress_level`: low / medium / high (순서형)
- `sleep_quality`: poor / average / good (순서형)
- `physical_activity_level`: sedentary / moderate / active (순서형)
- `smoking_alcohol`: no / occasional / yes (순서형)
- `gender`: male / female / other

## 결측치
거의 모든 컬럼에 결측 존재 (train 기준 약 1~12%). test도 비슷한 비율로 결측 발생 → 의도적으로 주입된 결측으로 추정(Playground 시리즈 공통 특징).

- 결측치 자체가 예측에 유의미한 신호일 수 있음 → 컬럼별 "결측 여부" 플래그 피처 추가 (전처리에서 반영 완료)
- 순서형 범주형 컬럼은 ordinal encoding, 나머지는 one-hot 인코딩 적용 (전처리에서 반영 완료)

## 전처리 (완료 — `notebooks/01_preprocessing.ipynb`)
1. **결측 플래그 생성**: 13개 피처 각각에 `{col}_isnull` 이진 컬럼 추가
2. **결측치 대치**: 수치형 → train median, 범주형 → `"missing"` 카테고리로 대치. 기준값은 train으로만 계산해 test에 동일 적용 (data leakage 방지)
3. **인코딩**
   - 순서형 4개(`stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`) → `OrdinalEncoder`로 순서 보존
   - 명목형 2개(`diet_type`, `gender`) → one-hot (train 컬럼 기준으로 test reindex)
4. **타겟 인코딩**: `health_condition` → LabelEncoder (0=at-risk, 1=fit, 2=unhealthy)
5. **저장**: `processed/train_processed.csv`, `processed/test_processed.csv` — 결측치 0건 확인 완료

> 주의: 인코딩 후 컬럼명 충돌 이슈가 있었음 — `{col}_missing` 형태의 결측 플래그가 `diet_type`/`gender`의 `"missing"` 카테고리 원-핫 컬럼(`diet_type_missing` 등)과 이름이 겹쳐서 `.1` suffix가 붙는 문제 발생. 결측 플래그 접미사를 `_isnull`로 바꿔서 해결함.

## EDA 결과 (완료 — `notebooks/02_eda.ipynb`)

**범주형 피처 — 압도적으로 강한 신호**
- `stress_level`이 가장 강력한 예측 변수: `medium`이면 at-risk **99.4%**, `low`면 fit 20.1%, `high`면 unhealthy 27.9% (전체 평균 8.4%의 3배 이상)
- `physical_activity_level`도 매우 강함: `active`면 fit 17.2%(평균 대비 3배), `sedentary`/`moderate`는 fit이 거의 0%
- `sleep_quality`: `poor`면 unhealthy 13.6%, `good`이면 fit 8.2%/unhealthy 3.0%로 뚜렷한 대비
- `smoking_alcohol`: `yes`면 unhealthy 11.2%, `no`면 fit 7.9% — 약~중간 신호
- `diet_type`, `gender`는 타겟과 거의 무관 → 예측 기여도 낮을 가능성

**수치형 피처 — 뚜렷하지만 범주형보다 약한 신호**
- `fit` 그룹이 확연히 다름: `sleep_duration` 평균 7.95(vs 7.09/5.37), `step_count` 11,651(vs 8,407/8,670), `exercise_duration` 50.0(vs 38.0/39.0), `bmi` 21.83(가장 낮음)
- `heart_rate`, `water_intake`는 클래스 간 차이 거의 없음 → 예측 기여도 낮을 가능성

**상관관계**: 수치형 피처 간 상관계수 전부 0에 가까움 → 다중공선성 없음, 피처 제거/차원축소 불필요

**결측 여부 ↔ 타겟**: 대부분 `_isnull` 플래그는 신호 미미(차이 <1%p). 유일한 예외는 **`bmi`**(결측 시 unhealthy 2.9%/fit 8.2% vs 비결측 시 8.5%/5.7%) → `bmi_isnull`은 유지 가치 있음

## ⭐ 핵심 발견 — 라벨 생성 규칙 역추적 (팀원 분석, 직접 재현·검증 완료)

합성 데이터라는 점에 착안해 "타겟이 규칙으로 만들어졌을 것"이라 가정하고, 얕은 결정트리(`max_depth=4`)로 라벨 생성 규칙 자체를 역추적한 발견. `sleep_duration`, `stress_level`, `physical_activity_level`이 모두 관측된 행(511,675개, 74.1%)에서 아래 코드로 직접 재현·검증 완료 — 팀원 문서의 모든 수치와 소수점까지 일치.

```python
def branch(row):
    sd = row["sleep_duration"]
    if sd < 6:
        return "unhealthy" if row["stress_level"] == "high" else "at-risk"
    elif sd >= 7:
        if row["stress_level"] == "low" and row["physical_activity_level"] == "active":
            return "fit"
        return "at-risk"
    return "at-risk"  # 6 <= sleep_duration < 7 구간은 항상 at-risk
```

**브랜치별 순도 (완전관측 행 기준, 검증됨)**
| 브랜치 | 조건 | 행 수 | 비율 | at-risk | fit | unhealthy |
|---|---|---|---|---|---|---|
| A | 수면<6 & 스트레스 high | 41,677 | 8.1% | 1.20% | 0.30% | **98.50%** |
| B | 수면<6 & 그 외 | 70,540 | 13.8% | **99.40%** | 0.19% | 0.41% |
| C | 수면≥7 & 스트레스 low & 활동 active | 28,247 | 5.5% | 0.69% | **99.05%** | 0.26% |
| D | 나머지(6~7 구간 포함) | 371,211 | 72.5% | **99.25%** | 0.36% | 0.39% |

이 규칙만으로 완전관측 행에서 accuracy 0.99199, 규칙이 틀리는 행은 4,101개(0.80%)뿐. 이 틀린 행들을 나머지 6개 수치형 피처와 비교해도 유의미한 차이가 없음(전부 표준편차 0.1 미만) → **잔여 0.8%는 설명 불가능한 순수 노이즈**이며, `step_count`/`bmi` 등으로 파생 피처를 만들어도 여기서 더 짜낼 정보가 없다는 뜻.

**같은 결론을 뒷받침하는 증거들**
- `sleep_duration`의 클래스 비율이 정확히 6과 7(정수)에서 계단처럼 꺾임 — 생물학적 지표라면 있을 수 없는 패턴
- train/test의 컬럼별 결측률이 소수점 넷째 자리까지 일치 → 결측이 의도적으로 주입됨(MCAR), `_isnull` 플래그의 정보량은 실질적으로 0에 가까움 (앞서 EDA에서 `bmi_isnull`만 예외로 봤던 것도 재검토 필요 — 표본 크기 대비 우연일 가능성 있음)
- 대회 토론방(discussion/717222)에서도 동일한 규칙이 독립적으로 보고됨
- 원본 원천 데이터셋(college-student-health-behavior-dataset)에서는 이 규칙이 정확도 100%

**성능 상한 분석**: 이 3개 피처를 셀 단위로 이산화하고 각 셀에서 최적 결정을 내리면(사전확률 보정 기준) balanced accuracy **0.94124**가 나옴. 대회 1위 점수가 0.95085였으므로, **모델링·튜닝·앙상블·피처엔지니어링을 다 합쳐도 남은 여지는 약 +0.01 수준**. 즉 이 대회는 "더 좋은 모델을 만드는" 문제가 아니라 "① 결정규칙(사전확률 보정)을 정확히 세우고 ② 핵심 3피처가 결측된 행에서 대리변수로 최대한 복구하는" 문제에 가까움.

**결측 패턴별 성능 분해** (셀 단위 최적 결정 기준 BA)
| 결측 패턴 | 뜻 | 비율 | BA | 해석 |
|---|---|---|---|---|
| 완전관측 | - | 74.1% | 0.9676 | 이미 최적 |
| stress만 결측 | | 10.1% | 0.8771 | 개선 여지 |
| sleep만 결측 | | 9.2% | 0.8784 | 개선 여지 |
| activity만 결측 | | 4.2% | 0.9359 | 개선 여지 |
| 2개 이상 결측 | | ~2.3% | 0.56~0.81 | 거의 찍기 수준 |
| 3개 다 결측 | | 0.07% | 0.3333 | 원리적으로 불가능 |

**결측 행 대리변수 복구 가능성**
- `physical_activity_level` ← `step_count`: **복구 잘 됨** (step_count 상/중/하위 1/3이 active/moderate/sedentary와 거의 대응)
- `sleep_duration<6` ← `sleep_quality`: **부분 복구** (poor일 때 P(sleep<6)=0.319 vs 평균 0.195, 확률은 크게 움직이지만 hard label 반전은 적음 — **정확도가 아니라 확률/BA 관점으로 평가해야 함**, 이 대회의 핵심 함정과 동일한 패턴)
- `stress_level` ← 뚜렷한 대리변수 없음 → **이게 이 대회의 실질적인 벽**

## Balanced Accuracy 최적 결정규칙 — 핵심 함정 (팀원 분석)

- accuracy 최대화 규칙: $\arg\max_c P(c\mid x)$
- **balanced accuracy 최대화 규칙**: $\arg\max_c \dfrac{P(c\mid x)}{P(c)}$ — 각 클래스 확률을 그 클래스의 base rate(사전확률)로 나눠서 비교해야 함

즉 단순히 argmax 확률로 예측하면(다수결) accuracy는 0.9623으로 높아도 balanced accuracy는 0.8451에 그침(소수 클래스 recall이 낮아서). 반대로 사전확률로 나눠서 예측하면 accuracy는 0.9275로 떨어져도 balanced accuracy는 0.9412로 오름 — **accuracy와 balanced accuracy가 반대로 움직일 수 있다는 것이 이 대회의 핵심 함정.**

> ⚠️ `class_weight='balanced'`로 학습(사전 보정)한 모델에 예측 후 사전확률 보정을 또 적용하면 **이중 보정**되어 점수가 오히려 떨어짐. 학습 중 보정(class_weight) **또는** 학습 후 보정(prior division) 중 **하나만** 적용할 것.

## CV + 사전확률 보정 + 결측 복구 결과 (완료 — `notebooks/03_cv_and_prior_correction.ipynb`)

팀원 제안(1~3순위)을 그대로 실행하고 5-fold OOF로 검증함. 커널: `teammate` conda 환경(lightgbm 4.6.0) 등록.

**1) 고정 CV 폴드**: `StratifiedKFold(5, shuffle=True, random_state=42)`로 생성해 `processed/cv_folds.csv`(id, fold)에 저장 — 앞으로 모든 실험은 이 폴드로 통일

**2) 결측 복구 피처 추가**
- `physical_activity_level_recovered`: 결측일 때 `step_count`를 train 기준 3분위(경계 6,561 / 11,029)로 나눠 sedentary/moderate/active로 대치
- `sleep_duration_recovered`: 결측일 때 `sleep_quality`별 조건부 중앙값으로 대치 (poor→6.45, average→6.99, good→7.54)
- 결측 플래그는 핵심 3피처(`stress_level`, `sleep_duration`, `physical_activity_level`)만 유지 (EDA에서 나머지는 신호 미미하다고 확인됨)

**3) 4가지 방식 CV 비교 (OOF, 690,088행 전체)**
| 방식 | accuracy | balanced accuracy |
|---|---|---|
| A. plain argmax (보정 없음) | 0.96706 | 0.87602 |
| **B. 사전확률 보정** | 0.93922 | **0.94980** ← 채택 |
| C. class_weight='balanced' | 0.94212 | 0.94919 |
| D. 이중보정(대조군: C 위에 사전확률 보정까지 적용) | 0.85981 | 0.92517 |

- B와 C가 거의 동률이라 **사전확률 보정(B)**을 최종 전략으로 채택
- **팀원이 경고한 이중보정 함정이 실제로 재현됨**: D가 B/C보다 확연히 낮음 → class_weight와 사전확률 보정은 반드시 하나만 쓸 것
- 팀원이 계산한 이론적 상한(3피처 셀 단위 이산화, BA 0.94124)을 **저희 결과(0.94980)가 오히려 상회** — `sleep_duration`을 연속값 그대로 사용하고 결측 복구 피처를 추가한 효과로 추정. 결측 복구 피처 엔지니어링이 실제로 유효했다는 뜻

**4) Feature Importance — split count vs gain 비교**

기본 LightGBM importance(`split` = 분기에 사용된 횟수)로 보면 `bmi`, `sleep_duration_recovered`, `water_intake`, `heart_rate` 등 연속형 피처가 상위권으로 나오는데, 이건 **연속형 피처는 트리가 여러 번 나눠 쓸 수 있어서 분기 횟수 자체가 많아지는 착시**입니다(각 분기의 실제 정보 이득은 작을 수 있음). 분기당 정보 이득을 반영하는 `gain` 기준으로 다시 보면 EDA·라벨 규칙 분석과 정확히 일치하는 그림이 나옵니다.

| 피처 | gain importance |
|---|---|
| `stress_level` | 2,419,875 |
| `sleep_duration_recovered` | 1,955,038 |
| `physical_activity_level_recovered` | 911,290 |
| `bmi` | 263,051 |
| `sleep_duration_isnull` | 167,002 |
| (이하 전부 10만 미만) | ... |
| `diet_type_*`, `gender_*` | 400~4,000 (거의 0 수준) |

→ 라벨 생성 규칙에 쓰인 3개 피처가 gain의 대부분을 차지, `diet_type`/`gender`는 3자리 수 수준으로 사실상 무의미함이 재확인됨. **모델 성능을 분석할 때는 반드시 gain(또는 permutation importance)을 봐야 하고, split count만 보면 잘못된 결론을 낼 수 있다는 것도 이번에 배운 점.**

**5) 산출물**: `processed/submission_v1.csv` (사전확률 보정 예측, 클래스 비율 at-risk 81.1%/unhealthy 11.5%/fit 7.4% — train 실제 비율 85.9/8.4/5.8과 다른 건 정상. BA를 최적화하려고 일부러 소수 클래스 쪽으로 예측을 옮긴 결과)

## LightGBM 하이퍼파라미터 튜닝 결과 (완료 — `notebooks/04_lgbm_tuning.ipynb`)

Optuna(TPE sampler, 30 trials)로 `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`를 탐색 (탐색 자체는 fold 0 하나로 빠르게, 최적 후보만 5-fold 전체로 재검증). 사전확률 보정 결정규칙은 그대로 유지.

| | balanced accuracy |
|---|---|
| baseline (03, 튜닝 전) | 0.94980 |
| tuned (04, Optuna 최적 파라미터, 5-fold 재검증) | 0.94987 |
| **개선폭** | **+0.00007** |

**최적 파라미터**: `learning_rate=0.0478`, `num_leaves=19`, `max_depth=9`, `min_child_samples=172`, `subsample=0.893`, `colsample_bytree=0.732`, `reg_alpha=0.0005`, `reg_lambda=0.323`

→ **가이드에서 예상했던 대로 튜닝의 효과는 사실상 없음(+0.0001 미만)**. 이미 사전확률 보정 + 결측 복구 피처만으로 이론적 상한(팀원 분석 기준 0.941~0.951 부근)에 근접했기 때문에, 트리 개수·깊이·정규화 같은 하이퍼파라미터를 아무리 조정해도 더 짜낼 게 거의 없다는 뜻. **"모델을 더 잘 튜닝하는 것"은 이 대회에서 시간 대비 효율이 가장 낮은 작업이라는 게 실험으로 재확인됨.**

산출물: `processed/submission_v2_tuned.csv` (현재 최종 제출 후보, 클래스 비율 at-risk 81.0%/unhealthy 11.6%/fit 7.4%)

## `stress_level` 예측 대치 실험 — 기각 (완료 — `notebooks/05_stress_level_imputation.ipynb`)

병목("stress_level은 대리변수가 없다")을 뚫어보려고, 관측된 `stress_level`로 보조 LightGBM 분류기(다른 모든 피처 사용)를 학습해서 `low/medium/high` 확률을 예측하고 이를 메인 모델에 피처 3개(`stress_proba_*`)로 추가하는 실험을 진행. 리키지 방지를 위해 04와 같은 5-fold를 그대로 재사용해 OOF 방식으로 확률 생성.

| 지표 | 값 |
|---|---|
| 보조 분류기 accuracy | 0.4609 |
| 최빈 클래스만 찍었을 때 baseline | 0.4311 |
| 보조 분류기 log_loss | 1.045 (완전 무작위 균등분포 기준 1.099와 거의 차이 없음) |
| 메인 모델 CV balanced accuracy (stress_proba 추가) | **0.94955** |
| 04 baseline 대비 | **-0.00025 (악화)** |

feature importance에서는 `stress_proba_high`가 2위로 높게 나왔지만(모델이 열심히 사용은 함), 정작 CV 성능은 오히려 떨어짐 — 신호가 baseline 대비 겨우 +3%p 수준으로 너무 약해서, 모델이 여기에 약간 과적합되며 노이즈를 추가한 것으로 추정.

→ **가설 기각**. 여러 피처를 동시에 조합해도 `stress_level`을 유의미하게 복구할 수 없다는 게 실험으로 확인됨 — 팀원의 원래 분석("이게 이 대회의 실질적인 벽")이 맞았음. **최종 제출은 `submission_v2_tuned.csv`(0.94987)를 그대로 유지.** 동일한 접근을 다시 시도할 필요는 없음.

## `sleep_duration` 회귀 기반 정밀 대치 실험 — 애매한 결과 (완료 — `notebooks/06_sleep_duration_regression.ipynb`)

기존(03/04) 방식은 `sleep_duration` 결측을 `sleep_quality`별 조건부 중앙값 하나로만 채워서(poor→6.45, average→6.99, good→7.54) 같은 `sleep_quality`인 사람은 전부 같은 값으로 뭉개졌음. 이번엔 `heart_rate`/`bmi`/`calorie_expenditure`/`step_count`/`exercise_duration`/`water_intake`/`stress_level`/`physical_activity_level_recovered`/`smoking_alcohol`/`diet_type`/`gender`까지 전부 써서 보조 LightGBM 회귀모델로 연속값을 예측(05와 동일하게 5-fold OOF로 리키지 방지).

**대치 자체의 품질은 확실히 개선됨** (관측된 `sleep_duration` 행에서 held-out 검증):
| 지표 | 회귀모델 | 기존(조건부 중앙값) | 개선폭 |
|---|---|---|---|
| RMSE | 1.1198 | 1.1342 | **+0.0145** |
| 6/7 경계 판정 정확도 | 49.38% | 42.79% | **+6.6%p** |

**그런데 메인 모델 CV 결과는 오히려 미세 하락**:
| | balanced accuracy |
|---|---|
| 04 baseline | 0.94987 |
| 06 회귀 대치 적용 | 0.94974 |
| 개선폭 | **-0.00013** |

sleep_duration 결측 행만 따로 본 BA: 0.86094.

→ **채택하지 않음(노이즈 수준)**. 대치 품질 자체는 분명히 좋아졌는데 최종 balanced accuracy에는 반영되지 않음 — 이미 성능 상한 근처(04 튜닝 실험의 +0.00007과 같은 자릿수)라 이 정도 개선으로는 최종 점수를 못 움직이는 것으로 보임. **중간 지표(RMSE 등)가 좋아졌다고 최종 지표가 반드시 좋아지는 건 아니라는 걸 배운 실험** — 트리 모델이 단순한 소수의 값(3~4개 값)에서 분기점을 안정적으로 찾는 것과, 정교한 연속값에서 분기점을 찾는 것 사이의 트레이드오프로 추정. **최종 제출은 `submission_v2_tuned.csv`(0.94987)를 그대로 유지.**

## Interaction 피처 + missing_count + Native NaN 실험 — 전부 기각 (완료 — `notebooks/07_interaction_and_native_nan.ipynb`)

팀원의 XGBoost 리포트(Balanced Accuracy 0.95024)를 참고해서 아이디어 3개를 04 baseline(0.94987) 위에 각각/결합해서 테스트했다. 04와 동일한 튜닝 파라미터·5-fold·사전확률 보정을 유지하고 Feature Set만 교체.

| Feature Set | 피처 수 | CV balanced accuracy | 04 대비 |
|---|---|---|---|
| **V0 baseline (04 재현)** | 22 | **0.94988** | (기준, 재현 오차 범위) |
| V1 +interaction (`sleep_lt6`, `high_stress_short_sleep`, `rule_branch` 등 라벨 규칙 명시적 주입) | 38 | 0.94964 | -0.00024 |
| V2 +missing_count (`core_missing_count`, `all_missing_count`) | 24 | 0.94977 | -0.00010 |
| V3 +native NaN (나머지 수치형을 median 대신 NaN 그대로 둬서 LightGBM 네이티브 결측 분기에 맡김) | 22 | 0.94977 | -0.00010 |
| V4 +전부 결합 | 40 | 0.94970 | -0.00017 |

**셋 다 성능을 오히려 소폭 깎아먹었다 (전부 노이즈 수준이지만 개선은 아님).**

- **Interaction 피처**: `stress_level`, `sleep_duration_recovered`, `physical_activity_level_recovered`를 원본 그대로 넣어주면 LightGBM이 이미 스스로 임계값·조합을 찾아서 gain importance 1~3위를 차지하고 있었음(04 결과). 여기에 같은 정보를 다르게 포장한 컬럼(`rule_branch`, `high_stress_short_sleep` 등)을 추가하는 건 새 정보가 아니라 **중복 정보**였고, `colsample_bytree=0.73`처럼 매 트리마다 피처를 무작위 샘플링하는 상황에서 분기 후보끼리 서로 자리를 뺏는 노이즈로 작용한 것으로 추정.
- **missing_count**: 이미 있는 개별 `_isnull` 플래그 3개와 정보가 겹쳐서 추가 신호가 거의 없었음.
- **Native NaN**: 처리 방식을 바꾼 컬럼들(`heart_rate`, `water_intake` 등)이 애초에 신호가 약했던 피처들이라(04 gain importance 하위권) 처리 방식 변경의 영향 자체가 미미했음.

→ **05, 06에 이어 3번째로 "baseline 대비 추가 피처가 도움이 안 되거나 해로웠다"는 패턴이 반복 확인됨.** 팀원의 XGBoost 파이프라인에서 효과가 있었던 아이디어라도, 이미 원본 피처만으로 핵심 신호를 다 활용하고 있는 저희 LightGBM 파이프라인에는 그대로 전이되지 않았음. **최종 제출은 `submission_v2_tuned.csv`(0.94987)를 계속 유지.** (노트북이 V0를 최고 성능으로 인식해 `submission_v5_interaction.csv`를 저장했으나, baseline과 피처 구성이 동일한 사실상 중복 파일 — 새로운 채택 아님.)

## 다음 단계
`stress_level`/`sleep_duration` 복구, interaction/missing_count/native NaN 추가까지 총 5개 아이디어가 전부 유의미한 개선을 못 만들었으므로, 남은 개선 여지는 사실상 한 가지로 좁혀짐:
1. **결측 2개 이상 겹친 행 처리** (약 2.3%, 이산화 기준 BA 0.56~0.81로 가장 취약한 구간) — 이 구간만 따로 떼어 분석/전용 전략 검토. 남은 개선 여지가 가장 많이 몰려있는 유일한 곳 (다만 표본이 작아 기대 효과는 제한적)
2. (완료, 효과 미미) 모델 다양성/앙상블, 하이퍼파라미터 튜닝 — 실험으로 +0.0001 미만 확인됨
3. (완료, 기각) `stress_level` 예측 대치 — 실험으로 -0.00025 확인됨, 재시도 불필요
4. (완료, 기각) `sleep_duration` 회귀 기반 정밀 대치 — 대치 품질은 개선됐으나 최종 CV는 -0.00013로 노이즈 수준, 재시도 불필요
5. (완료, 기각) Interaction 피처 / missing_count / native NaN — 셋 다 -0.0001~-0.0002 수준으로 하락, 재시도 불필요
6. (완료) `diet_type`/`gender`/`heart_rate`/`water_intake`는 gain importance로도 무의미함이 재확인됨 — 추가 피처엔지니어링 불필요

**현재까지 결론**: `submission_v2_tuned.csv`(CV balanced accuracy 0.94987)가 사실상의 실질적 상한으로 보임. 지금까지 시도한 5개의 개선 아이디어(예측 대치 2개, 피처 엔지니어링 3개)가 전부 실패했다는 것 자체가, 이 파이프라인이 "원본 피처 + 최소한의 결측 복구 + 올바른 결정규칙"만으로 이미 국소 최적점에 도달했다는 강한 증거임. 결측 2개 이상 겹친 행(전체의 2.3%, 약 15,900행)을 파고들어도 표본 자체가 작아서 기대 개선폭은 크지 않을 가능성이 높음 — 이 대회는 "더 나은 모델/피처"보다 "라벨 생성 규칙을 정확히 찾아내고 사전확률 보정을 올바르게 적용하는 것"이 점수의 대부분을 결정한다는 게 여러 번의 실험으로 재확인됨.

## 출처
- Yao Yan, Walter Reade, Elizabeth Park. Predicting Student Health Risk. https://kaggle.com/competitions/playground-series-s6e7, 2026. Kaggle.
- 원본 데이터셋: https://www.kaggle.com/datasets/ziya07/college-student-health-behavior-dataset
- 라벨 규칙 역추적 토론: https://www.kaggle.com/competitions/playground-series-s6e7/discussion/717222
- CV vs LB 관계 분석: https://www.kaggle.com/competitions/playground-series-s6e7/discussion/718258
- 팀원 EDA 노트북: `notebooks/01_EDA.ipynb`
