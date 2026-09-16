# 1. 최종 모델 및 성능

> 라벨 생성 규칙 기반 결측 복구 Feature Engineering
>
> - 사전확률 보정(Prior-corrected) 결정규칙
> - Hyperparameter Tuned LightGBM

**최종 성능**

> **Balanced Accuracy = 0.94987**

Target은 다음 3개 클래스로 구성된다.

| Class | 비율 |
| --- | --- |
| at-risk | 약 85.9% |
| unhealthy | 약 8.4% |
| fit | 약 5.8% |

클래스 불균형이 매우 심하기 때문에 일반 Accuracy보다 **Balanced Accuracy**를 주요 평가지표로 사용하였다.

---

# 2. EDA에서 확인한 핵심 패턴

EDA와 라벨 생성 규칙 역추적(팀원 분석, `max_depth=4` 결정트리로 재현) 결과, 단순한 개별 변수 효과보다 **`sleep_duration` × `stress_level` × `physical_activity_level` 조합과 threshold에서 Target이 거의 결정적으로 구분되는 현상**이 확인되었다.

### Sleep Duration

수치형 변수 가운데 `health_condition`을 가장 잘 구분하는 변수로 확인되었다. 단, 선형적으로 작용하기보다는 **6시간과 7시간 부근에서 강한 threshold effect**가 존재했다 — 클래스 비율이 정확히 정수 지점에서 계단처럼 꺾이는 패턴으로, 실제 생물학적 지표라면 나타날 수 없는 형태였다.

### Stress Level × Sleep Duration

완전관측 행(511,675개, 74.1%) 기준으로 재현한 결과:

| 조건 | 행 수 | at-risk | fit | unhealthy |
| --- | --- | --- | --- | --- |
| sleep < 6h & stress = high | 41,677 | 1.20% | 0.30% | **98.50%** |
| sleep < 6h & stress ≠ high | 70,540 | **99.40%** | 0.19% | 0.41% |

> **High Stress + Sleep < 6h → Unhealthy (약 98.5%)**

라는 매우 강한 비선형 패턴이 존재한다.

### Low Stress × Sleep × Activity

| 조건 | 행 수 | at-risk | fit | unhealthy |
| --- | --- | --- | --- | --- |
| sleep ≥ 7h & stress = low & activity = active | 28,247 | 0.69% | **99.05%** | 0.26% |
| 그 외(6~7h 구간 포함) | 371,211 | **99.25%** | 0.36% | 0.39% |

> **Low Stress + Sleep ≥ 7h + Active → Fit (약 99.05%)**

단순히 스트레스가 낮은 것만으로는 부족하고, **수면시간과 활동 수준이 동시에 조건을 만족해야** `fit`으로 강하게 갈렸다.

### 결측 패턴별 성능 분해

이 3개 핵심 피처를 이산화해 셀 단위 최적 결정을 내리면 balanced accuracy 상한은 **0.9412**(대회 1위 0.95085)로 계산되었다. 결측 조합별로 보면:

| 결측 패턴 | 비율 | BA |
| --- | --- | --- |
| 완전관측 | 74.1% | 0.9676 |
| stress만 결측 | 10.1% | 0.8771 |
| sleep만 결측 | 9.2% | 0.8784 |
| activity만 결측 | 4.2% | 0.9359 |
| 2개 이상 결측 | ~2.3% | 0.56~0.81 |

→ **남은 개선 여지는 대부분 "핵심 3피처가 결측된 행을 어떻게 다루는가"에 몰려 있다**는 것이 이후 Feature Engineering 방향을 결정했다.

---

# 3. 통계 분석 결과와 Feature Engineering 연결

EDA 및 라벨 규칙 역추적 결과를 종합하면 핵심 변수의 우선순위는 다음과 같다 (LightGBM gain importance 기준, 4절 참고).

| 중요도 | 주요 변수 |
| --- | --- |
| 매우 높음 | `stress_level`, `sleep_duration` |
| 높음 | `physical_activity_level` |
| 중간 | `bmi`, `sleep_duration_isnull` |
| 낮음 | `exercise_duration`, `step_count`, `water_intake`, `calorie_expenditure`, `heart_rate` |
| 단독 효과 거의 없음 | `diet_type`, `gender`, `smoking_alcohol` |

이를 바탕으로 **인위적인 polynomial/interaction feature를 대량 생성하는 대신, EDA에서 확인된 임계값(6h/7h)과 상호작용을 트리 기반 모델(LightGBM)이 분기(split)로 직접 학습하도록 원자 피처를 그대로 넘기고, 대신 결측 처리에 Feature Engineering 역량을 집중**하였다. Gradient Boosting Tree는 `sleep_duration < 6`, `sleep_duration >= 7` 같은 임계값을 스스로 분기점으로 찾아낼 수 있기 때문에, 별도의 `sleep_lt6`, `high_stress_short_sleep` 같은 수동 interaction dummy 없이도 gain importance에서 핵심 3피처가 상위권을 차지하는 것을 확인하였다(4절).

---

# 4. Feature Engineering

## A. 결측 복구 피처 (Recovery Features)

핵심 3피처 중 대리변수가 있는 2개만 복구하였다.

```python
# physical_activity_level 결측 -> step_count 3분위 매핑
step_tertiles = train["step_count"].quantile([1/3, 2/3]).values  # [6561, 11029]

data["physical_activity_level_recovered"] = pd.cut(
    data["step_count"],
    bins=[-np.inf, step_tertiles[0], step_tertiles[1], np.inf],
    labels=["sedentary", "moderate", "active"],
)
# physical_activity_level이 관측된 행은 원래 값 유지, 결측 행만 위 proxy로 대치
# step_count까지 결측인 극소수 행은 "moderate"로 대치

# sleep_duration 결측 -> sleep_quality별 조건부 중앙값
sleep_quality_cond = {"poor": 6.45, "average": 6.99, "good": 7.54, "missing": 6.99}
data["sleep_duration_recovered"] = data["sleep_duration"].fillna(
    data["sleep_quality"].map(sleep_quality_cond)
)
```

`stress_level`은 EDA에서 뚜렷한 대리변수가 확인되지 않아 복구하지 않고 `"missing"` 카테고리로 유지하였다 (6절 실험 참고).

## B. Missing Indicator

핵심 3피처에 대해서만 결측 플래그를 생성하였다.

```python
for col in ["stress_level", "sleep_duration", "physical_activity_level"]:
    data[f"{col}_isnull"] = data[col].isna().astype("int8")
```

나머지 컬럼의 결측 플래그는 EDA에서 타겟과의 연관성이 거의 없는 것으로 확인되어(결측/비결측 그룹 간 타겟 비율 차이 1%p 미만) 제외하였다.

## C. 시도했으나 채택하지 않은 Feature Engineering

아래 두 가지는 실제로 구현·검증했으나 최종 파이프라인에는 포함하지 않았다 (6절에서 상세).

- `stress_level` 예측 대치: 다른 모든 피처로 보조 분류기를 학습해 확률 3개를 피처로 추가 → CV balanced accuracy 하락(-0.00025)
- `sleep_duration` 회귀 기반 정밀 대치: `sleep_quality` 대신 `bmi`/`exercise_duration`/`step_count` 등 전체 피처로 연속값 회귀 예측 → 대치 자체는 개선됐으나 최종 CV는 하락(-0.00013)

---

# 5. Feature Ablation (실험 단계별 비교)

동일한 고정 5-fold(`StratifiedKFold(5, shuffle=True, random_state=42)`)로 각 실험을 비교하였다.

| 실험 | Feature/전략 구성 | CV Balanced Accuracy | 채택 |
| --- | --- | --- | --- |
| 03 (baseline) | 원본 + 결측 복구 피처 2개 + 결측 플래그 3개 + 사전확률 보정 | 0.94980 | 기준선 |
| 04 | baseline + Optuna 하이퍼파라미터 튜닝 | **0.94987** | ✅ 최종 채택 |
| 05 | baseline + `stress_level` 예측 대치 피처 3개 | 0.94955 | ❌ 기각 |
| 06 | baseline + `sleep_duration` 회귀 정밀 대치 | 0.94974 | ❌ 기각 |
| 07-V1 | baseline + interaction 피처(threshold dummy, `rule_branch` 등) | 0.94964 | ❌ 기각 |
| 07-V2 | baseline + `missing_count` 집계 피처 | 0.94977 | ❌ 기각 |
| 07-V3 | baseline + native NaN(수치형 median 미대치) | 0.94977 | ❌ 기각 |
| 07-V4 | baseline + 07-V1~V3 전부 결합 | 0.94970 | ❌ 기각 |
| 08 | 04 + FT-Transformer 블렌딩(alpha=0.95) | 0.94994 | ❌ 기각(노이즈 수준) |

> 이번 데이터에서는 **Feature/모델을 더 추가하는 것이 항상 도움이 되지 않았다** — 이미 raw feature + 최소한의 복구 피처만으로 트리 모델이 핵심 상호작용을 거의 다 학습했기 때문에(3절), 추가 피처나 다른 모델과의 앙상블이 오히려 노이즈로 작용하거나 무의미했던 사례가 총 7건(05, 06, 07-V1~V4, 08) 확인되었다. 최종 Feature Set/모델은 **04(baseline + 복구 피처 2개, LightGBM 단독)** 로 채택하였다.

---

# 6. 범주형 변수 처리

순서 정보가 있는 컬럼과 없는 컬럼을 구분해서 인코딩하였다.

```python
from sklearn.preprocessing import OrdinalEncoder

# 순서형 -> OrdinalEncoder (순서 보존)
ORDINAL_COLS = {
    "stress_level": ["low", "medium", "high"],
    "sleep_quality": ["poor", "average", "good"],
    "physical_activity_level_recovered": ["sedentary", "moderate", "active"],
    "smoking_alcohol": ["no", "occasional", "yes"],
}
for col, order in ORDINAL_COLS.items():
    encoder = OrdinalEncoder(categories=[order + ["missing"]])
    train[col] = encoder.fit_transform(train[[col]])
    test[col] = encoder.transform(test[[col]])

# 명목형 -> One-Hot Encoding
NOMINAL_COLS = ["diet_type", "gender"]
train_ohe = pd.get_dummies(train[NOMINAL_COLS], prefix=NOMINAL_COLS)
test_ohe = pd.get_dummies(test[NOMINAL_COLS], prefix=NOMINAL_COLS).reindex(
    columns=train_ohe.columns, fill_value=0
)
```

Train에서 Encoder를 학습하고 동일한 Encoder를 Test에 적용하였다 (data leakage 방지).

---

# 7. Class Imbalance 처리 — 결정규칙 비교

Target 분포가 심하게 불균형하므로, **학습 시점 보정(class_weight)** 과 **예측 시점 보정(사전확률 나누기)** 두 방식을 모두 구현해서 비교하였다.

```python
def prior_corrected_predict(proba, class_order, priors):
    """Balanced Accuracy 최대화 규칙: argmax P(c|x)/P(c)"""
    prior_arr = np.array([priors[c] for c in class_order])
    scores = proba / prior_arr
    return np.array(class_order)[scores.argmax(axis=1)]

train_priors = train["health_condition"].value_counts(normalize=True).to_dict()
# {'at-risk': 0.8587, 'unhealthy': 0.0836, 'fit': 0.0577}
```

**4가지 방식 비교 (OOF, 690,088행)**

| 방식 | accuracy | balanced accuracy |
| --- | --- | --- |
| A. plain argmax (보정 없음) | 0.96706 | 0.87602 |
| **B. 사전확률 보정** | 0.93922 | **0.94980** |
| C. `class_weight='balanced'` | 0.94212 | 0.94919 |
| D. 이중보정(B+C 동시 적용, 대조군) | 0.85981 | 0.92517 |

B와 C가 거의 동률이라 **사전확률 보정(B)**을 최종 결정규칙으로 채택하였다.

> ⚠️ **`class_weight='balanced'`와 사전확률 보정을 동시에 적용하면(D) balanced accuracy가 0.925까지 하락한다.** 학습 중 보정과 예측 후 보정은 반드시 하나만 사용해야 한다.

---

# 8. Validation Strategy

클래스 비율을 유지하기 위해 Stratified Split을 사용하였고, **팀 전원이 동일한 폴드를 재사용**하도록 별도 파일로 고정하였다.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train, train["health_condition"])):
    train.loc[val_idx, "fold"] = fold

train[["id", "fold"]].to_csv("processed/cv_folds.csv", index=False)
```

평가지표는 **Balanced Accuracy**(각 클래스 Recall의 평균)를 사용하였다.

---

# 9. LightGBM Baseline

```python
import lightgbm as lgb

baseline_params = dict(
    objective="multiclass", num_class=3, n_estimators=500,
    learning_rate=0.05, num_leaves=63, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbosity=-1,
)
```

이 파라미터로 사전확률 보정을 적용한 결과가 0.94980 (5-fold OOF).

---

# 10. Hyperparameter Tuning

Feature Set(baseline + 결측 복구 피처 2개)을 고정한 뒤 Optuna(TPE sampler)로 LightGBM 하이퍼파라미터를 탐색하였다. 690,088행 전체에 5-fold를 매 trial마다 도는 것은 비용이 커서, **탐색 자체는 fold 0 하나로 빠르게 스코어링하고, 최적 후보만 5-fold 전체로 재검증**하는 방식을 사용하였다 (30 trials).

### 탐색 공간

| Hyperparameter | 범위 |
| --- | --- |
| `learning_rate` | 0.01 ~ 0.1 (log scale) |
| `num_leaves` | 15 ~ 255 |
| `max_depth` | 3 ~ 12 |
| `min_child_samples` | 10 ~ 200 |
| `subsample` | 0.6 ~ 1.0 |
| `colsample_bytree` | 0.5 ~ 1.0 |
| `reg_alpha` | 1e-8 ~ 10.0 (log scale) |
| `reg_lambda` | 1e-8 ~ 10.0 (log scale) |

```python
import optuna

def objective(trial):
    params = dict(
        objective="multiclass", num_class=3, random_state=42, verbosity=-1,
        n_estimators=1000,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )
    return quick_score(params)  # fold 0 기준 사전확률 보정 balanced accuracy

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)
```

### 최적 파라미터

```python
best_params = {
    "learning_rate": 0.047792122422826176,
    "num_leaves": 19,
    "max_depth": 9,
    "min_child_samples": 172,
    "subsample": 0.8929201812783885,
    "colsample_bytree": 0.7318659835628288,
    "reg_alpha": 0.0005023614837892232,
    "reg_lambda": 0.32275087452118445,
}
```

---

# 11. Tuned Model 재검증 (5-Fold 전체)

Optuna에서 얻은 `best_params`를 바로 최종 채택하지 않고, 전체 데이터에 대해 5-fold 전체로 다시 검증하였다.

```
Fold 0 Quick Scoring (30 trials)
      ↓
Best Parameter Candidate
      ↓
전체 Train, 5-Fold Stratified CV 재검증
      ↓
baseline(0.94980) 대비 개선 확인
```

| | balanced accuracy |
| --- | --- |
| baseline (03, 튜닝 전) | 0.94980 |
| tuned (04, 5-fold 재검증) | **0.94987** |
| 개선폭 | +0.00007 |

이미 3절에서 확인한 이론적 상한(0.9412, 결측 없는 조건부 상한 0.9676) 근처에 도달해 있어, 하이퍼파라미터를 조정해도 개선폭은 사실상 미미했다.

**Feature Importance (gain 기준, 상위)**

| 피처 | gain |
| --- | --- |
| `stress_level` | 2,419,875 |
| `sleep_duration_recovered` | 1,955,038 |
| `physical_activity_level_recovered` | 911,290 |
| `bmi` | 263,051 |
| `sleep_duration_isnull` | 167,002 |
| `diet_type_*`, `gender_*` | 400 ~ 4,000 (사실상 무의미) |

라벨 생성 규칙에 실제로 쓰인 3개 피처가 gain의 대부분을 차지하는 것으로 확인되어, 3절의 EDA 결론과 정확히 일치하였다.

> ※ split-count 기준 importance로는 `bmi`, `water_intake` 등 연속형 피처가 상위로 잘못 나온다 — 연속형 피처는 트리가 여러 번 나눠 쓸 수 있어 분기 횟수 자체가 많아지는 착시이므로, feature importance는 반드시 **gain** 기준으로 확인해야 한다.

---

# 12. 추가 실험 — 결측치 Predictive Imputation (기각)

병목 구간(3절 결측 패턴 분해)을 뚫어보기 위해 두 가지 predictive imputation을 시도했으나 최종 채택하지 않았다.

### `stress_level` 예측 대치

`stress_level`이 관측된 행으로 보조 LightGBM 분류기(다른 모든 피처 사용)를 학습해 `low/medium/high` 확률을 예측하고, 이를 피처 3개로 메인 모델에 추가. 5-fold OOF 방식으로 리키지 방지.

| 지표 | 값 |
| --- | --- |
| 보조 분류기 accuracy | 0.4609 |
| 최빈 클래스 baseline | 0.4311 |
| 보조 분류기 log_loss | 1.045 (균등분포 1.099와 거의 차이 없음) |
| 메인 모델 CV balanced accuracy | 0.94955 (**-0.00025**) |

→ 여러 피처를 조합해도 `stress_level`엔 유의미한 신호가 거의 없음(+3%p뿐) → 가설 기각.

### `sleep_duration` 회귀 기반 정밀 대치

`sleep_quality` 하나가 아니라 `bmi`/`exercise_duration`/`step_count`/`calorie_expenditure` 등 전체를 활용한 LightGBM 회귀모델로 연속값 예측.

| 지표 | 회귀모델 | 기존(조건부 중앙값) |
| --- | --- | --- |
| RMSE | 1.1198 | 1.1342 |
| 6/7 경계 판정 정확도 | 49.38% | 42.79% |
| 메인 모델 CV balanced accuracy | 0.94974 | 0.94987 |

→ 대치 품질(RMSE, 경계판정)은 확실히 개선됐지만 최종 balanced accuracy에는 반영되지 않음(-0.00013, 노이즈 수준) → 기각.

두 실험 모두 **중간 지표 개선이 최종 지표 개선을 보장하지 않는다**는 것과, 이미 성능 상한 근처에서는 추가 정보가 새로운 신호가 아니라 노이즈로 작용할 수 있다는 것을 보여준 사례다.

---

# 12-2. 추가 실험 — Interaction / missing_count / Native NaN (기각)

다른 팀의 XGBoost 파이프라인(수동 threshold·interaction feature, missing indicator 다수, native NaN 처리)에서 착안해 동일 아이디어 3가지를 04 baseline 위에서 테스트하였다. 04와 동일한 튜닝 파라미터·5-fold·사전확률 보정을 유지하고 Feature Set만 교체.

```python
# interaction 피처 예시
data["sleep_lt6"] = (data["sleep_duration_recovered"] < 6).astype("int8")
data["high_stress_short_sleep"] = (data["stress_high"] & data["sleep_lt6"]).astype("int8")
data["rule_branch"] = ...  # 라벨 생성 규칙(A/B/C/D)을 단일 카테고리로 인코딩

# missing_count
data["core_missing_count"] = data[["stress_level_isnull", "sleep_duration_isnull",
                                     "physical_activity_level_isnull"]].sum(axis=1)
data["all_missing_count"] = data[ORIGINAL_FEATURE_COLS].isna().sum(axis=1)

# native NaN: 나머지 수치형을 median으로 채우지 않고 그대로 LightGBM에 전달
```

| Feature Set | 피처 수 | CV balanced accuracy | 04 대비 |
| --- | --- | --- | --- |
| V0 baseline (04 재현) | 22 | 0.94988 | (기준) |
| V1 +interaction | 38 | 0.94964 | -0.00024 |
| V2 +missing_count | 24 | 0.94977 | -0.00010 |
| V3 +native NaN | 22 | 0.94977 | -0.00010 |
| V4 +전부 결합 | 40 | 0.94970 | -0.00017 |

**세 아이디어 모두 baseline보다 낮았다.** 원인으로 추정되는 것:
- `stress_level`, `sleep_duration_recovered`, `physical_activity_level_recovered`를 원본 그대로 넣는 것만으로 LightGBM이 이미 gain importance 1~3위를 차지할 만큼 핵심 상호작용을 스스로 찾고 있었음(11절). 여기에 같은 정보를 재포장한 interaction/branch 컬럼을 더하는 것은 새 정보가 아니라 **중복 정보**였고, `colsample_bytree=0.73`처럼 매 트리마다 피처를 무작위 샘플링하는 상황에서 분기 후보 간 경쟁만 늘려 노이즈로 작용
- `missing_count`는 이미 있는 개별 `_isnull` 플래그 3개와 정보가 겹침
- `native NaN`으로 처리 방식을 바꾼 컬럼들은 애초에 gain importance 하위권으로 신호가 약한 피처들이라 영향 자체가 미미함

→ **05, 06에 이어 "baseline 대비 추가 피처가 도움이 안 됐다"는 패턴이 3번째로 반복 확인됨.** 다른 파이프라인(다른 모델·다른 인코딩 방식)에서 효과가 있었던 아이디어라도, 이미 원본 피처만으로 핵심 신호를 충분히 활용하고 있는 파이프라인에는 그대로 전이되지 않을 수 있다는 것을 보여준 사례. **최종 제출은 `submission_v2_tuned.csv`(0.94987)를 계속 유지.**

---

# 12-3. 추가 실험 — FT-Transformer + LightGBM 앙상블 (기각)

"Transformer 계열은 LightGBM과 모델 구조가 완전히 달라서 앙상블 시 에러가 분산되어 이득이 클 것"이라는 가설을 검증하기 위해, 정형 데이터용 Transformer(FT-Transformer 스타일: 수치형은 피처별 선형 토크나이저, 범주형은 임베딩, 전부 토큰화해서 self-attention에 입력)를 PyTorch로 직접 구현하고 04와 동일한 5-fold로 학습, LightGBM과 블렌딩을 시도하였다.

```python
class FTTransformerLite(nn.Module):
    def __init__(self, n_numeric, n_flags, cat_cardinalities, d_model=32, n_heads=4, n_layers=2):
        # 수치형/플래그: 피처별 학습되는 (weight, bias)로 토큰화
        # 범주형: nn.Embedding
        # [CLS] 토큰 + TransformerEncoder(2 layers) -> 분류 head
        ...
```

| | balanced accuracy |
| --- | --- |
| Transformer 단독 (5-fold OOF) | 0.94753 |
| LightGBM 단독 (04 재현) | 0.94988 |
| 블렌딩 최적 alpha(LightGBM 0.95 : Transformer 0.05) | 0.94994 |
| 04 baseline 대비 개선폭 | **+0.00007 (노이즈 수준)** |

**Transformer 단독 성능이 예상보다 훨씬 좋았다** — 처음 학습한 소규모 모델(2-layer, d_model=32)인데도 LightGBM과 0.0024 차이밖에 안 났다. `sleep_duration<6` 같은 날카로운 threshold를 트리보다 못 잡을 거라는 예상은 방향은 맞았지만 격차는 작았음 — 690k행이라는 데이터량 덕에 신경망도 threshold 함수를 상당히 잘 근사했다.

**그러나 앙상블 alpha를 0~1로 그리드서치한 결과가 거의 단조증가**하며 alpha=1(순수 LightGBM) 근처가 사실상 최적이었다 — 이득은 다른 노이즈 수준 실험들과 같은 자릿수(+0.00007)에 그쳤다. 원인: 라벨이 소수 피처의 명확한 규칙(2절)으로 생성돼 있어 LightGBM과 Transformer 둘 다 **같은 정답 함수를 근사**하는 셈이 되고, 둘 다 정확도가 비슷하면 실수하는 지점도 상당 부분 겹친다. "모델 계열이 다르면 앙상블이 항상 이득"이라는 일반론이, 신호가 뚜렷하고 이미 성능 상한 근처인 데이터에서는 잘 통하지 않는다는 것을 보여준 사례.

→ 최종 제출은 `submission_v2_tuned.csv`(0.94987)를 계속 유지.

---

# 13. 최종 학습

Feature Engineering과 Hyperparameter Tuning이 완료된 후 전체 Train 데이터로 최종 LightGBM 모델을 다시 학습하였다.

```python
final_model = lgb.LGBMClassifier(**best_params, objective="multiclass", num_class=3,
                                   n_estimators=500, random_state=42, verbosity=-1)
final_model.fit(train[FEATURE_COLS], train["target_enc"])
```

Test 데이터에도 Train과 동일한 결측 복구 규칙(step_count 3분위 경계, sleep_quality 조건부 중앙값은 train 기준으로 고정)과 Encoder를 적용하였다.

---

# 14. Test Prediction & Submission

```python
test_proba = final_model.predict_proba(test[FEATURE_COLS])
test_pred = prior_corrected_predict(test_proba, lgb_class_order, train_priors)

submission = pd.DataFrame({"id": test["id"], "health_condition": test_pred})
submission.to_csv("submission_v2_tuned.csv", index=False)
```

클래스 비율: at-risk 81.0% / unhealthy 11.6% / fit 7.4% — train 실제 비율(85.9/8.4/5.8)과 다른 것은 정상이다. Balanced Accuracy를 최적화하기 위해 의도적으로 소수 클래스 쪽으로 예측을 옮긴 결과다.

최종 제출 결과:

> **Balanced Accuracy = 0.94987**

---

# 15. 최종 Pipeline

```
Raw Train / Test Data
          │
          ▼
────────────────────────
        EDA
────────────────────────
          │
          ├─ Target Imbalance (85.9 / 8.4 / 5.8%)
          ├─ Missing Value 패턴 (MCAR, 컬럼별 1~12%)
          └─ 라벨 생성 규칙 역추적 (결정트리, max_depth=4)
                → sleep_duration × stress_level × physical_activity_level
          │
          ▼
────────────────────────
   Feature Engineering
────────────────────────
          │
          ├─ physical_activity_level_recovered ← step_count 3분위
          ├─ sleep_duration_recovered ← sleep_quality 조건부 중앙값
          └─ Missing Indicator ×3 (stress/sleep/activity)
          │
          ▼
────────────────────────
   Feature Ablation (03~06)
────────────────────────
          │
          ├─ 03: baseline + 복구 피처 2개        (0.94980)
          ├─ 04: + Optuna 튜닝                    (0.94987) ★
          ├─ 05: + stress_level 예측 대치         (0.94955, 기각)
          └─ 06: + sleep_duration 회귀 대치       (0.94974, 기각)
          │
          ▼
────────────────────────
      Preprocessing
────────────────────────
          │
          ├─ 순서형 → OrdinalEncoder
          └─ 명목형 → One-Hot Encoding
          │
          ▼
────────────────────────
     Class Imbalance
────────────────────────
          │
          ├─ A. plain argmax                (0.876)
          ├─ B. 사전확률 보정                (0.950) ★
          ├─ C. class_weight='balanced'      (0.949)
          └─ D. 이중보정(B+C)                (0.925, 금지)
          │
          ▼
────────────────────────
     LightGBM Baseline
────────────────────────
          │
          ▼
────────────────────────
 Hyperparameter Tuning
────────────────────────
          │
          ├─ Optuna TPE, 30 trials (fold 0 quick scoring)
          └─ best_params 확정
          │
          ▼
────────────────────────
 Stratified 5-Fold CV 재검증
────────────────────────
          │
          └─ OOF Balanced Accuracy = 0.94987
          │
          ▼
────────────────────────
 Full Train Retraining
────────────────────────
          │
          ▼
      Test Prediction
     (사전확률 보정 적용)
          │
          ▼
       Submission
          │
          ▼
 Balanced Accuracy
      = 0.94987
```

---

# 16. 최종 결론

이번 모델링에서는 인위적인 파생변수를 대량 생성하기보다 **EDA와 라벨 생성 규칙 역추적으로 확인된 임계값·상호작용을 LightGBM(Gradient Boosting Tree)이 스스로 분기로 학습하도록** 하고, Feature Engineering 역량은 **결측치 복구**에 집중하였다.

Feature Ablation(5절) 결과, 원본 피처 + 최소한의 결측 복구 피처 2개만으로 이미 이론적 상한(0.9412) 근처에 도달했으며, 여기에 추가 피처를 더하는 시도는 총 6가지(`stress_level` 예측 대치, `sleep_duration` 정밀 회귀 대치, interaction 피처, missing_count, native NaN, 이들의 결합) 모두 오히려 성능을 소폭 떨어뜨렸다. 심지어 완전히 다른 모델 계열(FT-Transformer)과의 앙상블(12-3절)조차 노이즈 수준(+0.00007)의 이득에 그쳤다 — Transformer 단독 성능(0.94753)은 예상보다 훨씬 좋았지만, 라벨이 소수 피처의 명확한 규칙으로 생성돼 있어 LightGBM과 결국 같은 정답 함수를 근사하게 되고, 그만큼 두 모델의 오답 패턴도 겹쳐서 앙상블 다양성 효과가 거의 없었다. 이는 데이터가 이미 소수의 명확한 규칙으로 생성된 합성 데이터라 원본 피처만으로 트리 모델이 핵심 신호를 충분히 포착했고, 추가 피처나 다른 모델과의 결합은 새 정보가 아니라 중복·노이즈로 작용했기 때문으로 판단된다.

클래스 불균형은 `class_weight`가 아니라 **예측 시점의 사전확률 보정**(`argmax P(c|x)/P(c)`)으로 처리하였고, 이 결정규칙 하나가 plain argmax 대비 +0.074라는 압도적인 개선을 만들어 이번 프로젝트에서 가장 큰 지렛대였다. `class_weight`와 사전확률 보정을 동시에 적용하는 이중보정은 명확히 해롭다는 것도 실험으로 확인하였다.

최종적으로 **EDA → 라벨 규칙 역추적 → 결측 복구 Feature Engineering → 결정규칙 비교 → Hyperparameter Tuning → Stratified 5-Fold CV → Full Training**의 파이프라인을 구축하였으며, 최종 **Balanced Accuracy 0.94987**을 달성하였다 (대회 1위 0.95085 대비 약 0.001 차이로, 3피처 기반 이론적 상한에 근접한 수준).
