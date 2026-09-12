# 최종 파이프라인 — Predicting Student Health Risk (S6E7)

이 문서는 최종 채택된 파이프라인만 순서대로 기록합니다. 실험 과정과 기각된 시도들의 근거는 `PROJECT_GUIDE.md`를 참고하세요.

**최종 제출 파일**: `playground-series-s6e7/processed/submission_v2_tuned.csv`
**CV balanced accuracy (5-fold OOF)**: **0.94987**
**재현 노트북**: `notebooks/04_lgbm_tuning.ipynb` (커널: **Python (teammate)**, lightgbm 4.6.0 + optuna 5.0.0)

---

## 0. 준비물

| 항목 | 내용 |
|---|---|
| 입력 데이터 | `playground-series-s6e7/train.csv` (690,088행), `test.csv` (295,753행) |
| 실행 환경 | conda 환경 `teammate` (pandas 2.2.3, scikit-learn 1.8.0, lightgbm 4.6.0, optuna 5.0.0) |
| 시드 | `SEED = 42` (전 단계 공통) |

---

## 1. 고정 CV 폴드

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train, train["health_condition"])):
    train.loc[val_idx, "fold"] = fold
```
- `processed/cv_folds.csv` (id, fold)로 저장해서 이후 모든 실험이 동일 폴드를 재사용
- `health_condition` 기준 층화(stratify) — 클래스 비율(85.9/8.4/5.8%)이 각 폴드에 동일하게 유지됨

## 2. 결측 복구 피처 (2개)

라벨 생성 규칙에 쓰이는 핵심 3피처(`stress_level`, `sleep_duration`, `physical_activity_level`) 중 대리변수가 있는 2개만 복구. `stress_level`은 대리변수가 없어 복구하지 않음(실험으로 확인됨, `PROJECT_GUIDE.md` 참고).

```python
# (1) physical_activity_level 결측 -> step_count 3분위 매핑
step_tertiles = train["step_count"].quantile([1/3, 2/3]).values  # [6561, 11029]
# 하위 1/3 -> sedentary, 중위 1/3 -> moderate, 상위 1/3 -> active
# step_count까지 없으면 "moderate"로 대치

# (2) sleep_duration 결측 -> sleep_quality별 조건부 중앙값
# poor -> 6.45, average -> 6.99, good -> 7.54, sleep_quality까지 결측 -> 전체 중앙값(6.99)
```

## 3. 결측 플래그 (3개만 유지)

```python
stress_level_isnull, sleep_duration_isnull, physical_activity_level_isnull  # 0/1
```
나머지 피처의 `_isnull` 플래그는 EDA에서 신호가 거의 없다고 확인되어 제외.

## 4. 나머지 결측치 처리

| 컬럼 종류 | 처리 |
|---|---|
| 나머지 수치형 (`heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake`) | train 기준 median으로 대치 |
| 나머지 범주형 (`stress_level`, `smoking_alcohol`, `diet_type`, `gender`, `sleep_quality`) | `"missing"`을 별도 카테고리로 취급 |

> 대치 기준값(median, 조건부 중앙값, step_count 3분위 경계)은 전부 **train으로만 계산**해서 test에 동일 적용 (data leakage 방지).

## 5. 인코딩

```python
# 순서형 -> OrdinalEncoder (순서 정보 보존)
ORDINAL_COLS = {
    "stress_level": ["low", "medium", "high"],
    "sleep_quality": ["poor", "average", "good"],
    "physical_activity_level_recovered": ["sedentary", "moderate", "active"],
    "smoking_alcohol": ["no", "occasional", "yes"],
}  # + "missing"을 마지막 카테고리로 추가

# 명목형 -> one-hot (pd.get_dummies), train 컬럼 기준으로 test reindex
NOMINAL_COLS = ["diet_type", "gender"]

# 타겟 -> LabelEncoder: at-risk=0, fit=1, unhealthy=2 (알파벳순)
```

**최종 피처 22개** = 수치형 7 + 순서형 인코딩 4 + 결측 플래그 3 + one-hot 8(diet_type 4 + gender 4)

## 6. 모델 — LightGBM (Optuna 튜닝 완료)

```python
import lightgbm as lgb

params = dict(
    objective="multiclass", num_class=3, random_state=42, verbosity=-1,
    n_estimators=500,
    learning_rate=0.047792122422826176,
    num_leaves=19,
    max_depth=9,
    min_child_samples=172,
    subsample=0.8929201812783885,
    colsample_bytree=0.7318659835628288,
    reg_alpha=0.0005023614837892232,
    reg_lambda=0.32275087452118445,
)
model = lgb.LGBMClassifier(**params)
model.fit(train[FEATURE_COLS], train["target_enc"])
```
- Optuna(TPE sampler, 30 trials)로 탐색, 5-fold 전체로 재검증
- 튜닝 전(03) 대비 개선폭은 +0.00007로 미미함 — 이미 이론적 상한 근처였기 때문 (`PROJECT_GUIDE.md`의 핵심 발견 섹션 참고)

## 7. 결정규칙 — 사전확률 보정 (핵심)

**절대 규칙: `class_weight='balanced'`와 아래 사전확률 보정을 동시에 쓰지 말 것 (이중보정 시 -0.02~0.07 하락 확인됨)**

```python
import numpy as np

def prior_corrected_predict(proba, class_order, priors):
    prior_arr = np.array([priors[c] for c in class_order])
    scores = proba / prior_arr          # P(c|x) / P(c)
    return np.array(class_order)[scores.argmax(axis=1)]

train_priors = train["health_condition"].value_counts(normalize=True).to_dict()
# {'at-risk': 0.8587, 'unhealthy': 0.0836, 'fit': 0.0577}

test_proba = model.predict_proba(test[FEATURE_COLS])
test_pred = prior_corrected_predict(test_proba, lgb_class_order, train_priors)
```
- Balanced accuracy를 최대화하려면 단순 `argmax P(c|x)`가 아니라 `argmax P(c|x)/P(c)`를 써야 함
- plain argmax만 쓰면 accuracy는 높아도(0.967) balanced accuracy는 낮음(0.876) — 다수 클래스 쏠림 때문

## 8. 제출 파일 생성

```python
submission = pd.DataFrame({"id": test["id"], "health_condition": test_pred})
submission.to_csv("processed/submission_v2_tuned.csv", index=False)
```
- 클래스 비율: at-risk 81.0% / unhealthy 11.6% / fit 7.4% (train 실제 비율 85.9/8.4/5.8과 다른 게 정상 — BA 최적화를 위해 의도적으로 소수 클래스 쪽으로 예측을 옮긴 결과)

> ⚠️ **`sample_submission.csv`를 그대로 제출하지 말 것** — 전 행이 `at-risk`로만 채워진 템플릿이라 balanced accuracy가 0.3333으로 나옴 (실제로 한 번 착각해서 제출했던 사례 있음).

---

## 성능 요약

| 단계 | balanced accuracy (5-fold OOF) |
|---|---|
| plain argmax (보정 없음) | 0.8760 |
| 사전확률 보정 (튜닝 전) | 0.9498 |
| **사전확률 보정 + Optuna 튜닝 (최종)** | **0.9499** |
| (참고) 대회 1위 | 0.95085 |
| (참고) 라벨 생성 규칙 3피처만으로 이산화한 이론적 상한 | 0.9412 |

## 시도했으나 채택하지 않은 것 (재시도 불필요)

| 시도 | 결과 | 노트북 |
|---|---|---|
| `stress_level` 예측 대치 (다른 피처로 보조 분류기 학습) | -0.00025, 신호 거의 없음(보조모델 accuracy 46.1% vs baseline 43.1%) | `05_stress_level_imputation.ipynb` |
| `sleep_duration` 회귀 기반 정밀 대치 | 대치 자체는 개선(RMSE, 경계판정↑)됐으나 최종 CV는 -0.00013로 노이즈 수준 | `06_sleep_duration_regression.ipynb` |
| 모델 다양성/앙상블, 추가 하이퍼파라미터 탐색 | 이론상 기대효과 +0.001 미만 | — |
