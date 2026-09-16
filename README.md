# Predicting Student Health Risk — Playground Series S6E7

Kaggle: https://www.kaggle.com/competitions/playground-series-s6e7

3-class 분류(`health_condition`: at-risk / unhealthy / fit) 문제를 EDA → 라벨 생성 규칙 역추적 → 결측치 복구 → LightGBM 모델링 순서로 진행한 프로젝트입니다.

**최종 결과**: CV Balanced Accuracy **0.94987** (LightGBM + 사전확률 보정 결정규칙)

## 문서
- [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) — EDA부터 모든 실험(성공/실패 포함)까지의 전체 로그
- [`LIGHTGBM_MODELING_RESULTS.md`](LIGHTGBM_MODELING_RESULTS.md) — LightGBM 모델링 결과 리포트
- [`PIPELINE.md`](PIPELINE.md) — 최종 채택 파이프라인 재현 코드

## 노트북
| 파일 | 내용 |
|---|---|
| `notebooks/01_preprocessing.ipynb` | 기본 전처리 |
| `notebooks/02_eda.ipynb` | EDA |
| `notebooks/03_cv_and_prior_correction.ipynb` | 고정 CV 폴드, 결측 복구 피처, 사전확률 보정 결정규칙 비교 |
| `notebooks/04_lgbm_tuning.ipynb` | Optuna 하이퍼파라미터 튜닝 (**최종 채택**) |
| `notebooks/05_stress_level_imputation.ipynb` | `stress_level` 예측 대치 실험 (기각) |
| `notebooks/06_sleep_duration_regression.ipynb` | `sleep_duration` 회귀 기반 정밀 대치 실험 (기각) |
| `notebooks/07_interaction_and_native_nan.ipynb` | Interaction/missing_count/Native NaN 실험 (기각) |
| `notebooks/08_transformer_ensemble.ipynb` | FT-Transformer + LightGBM 앙상블 실험 (기각) |
| `notebooks/09_missing_segment_specialist.ipynb` | 결측 2개 이상 세그먼트 전용 서브모델 실험 (기각, 최종) |

## 데이터 준비

Kaggle 대회 데이터는 재배포 제한 및 용량 문제로 이 저장소에 포함하지 않았습니다. 노트북을 실행하려면 아래 순서로 준비하세요.

1. [대회 페이지](https://www.kaggle.com/competitions/playground-series-s6e7/data)에서 `train.csv`, `test.csv`, `sample_submission.csv`를 다운로드
2. 저장소 루트에 다음 구조로 배치:
   ```
   playground-series-s6e7/
   ├── train.csv
   ├── test.csv
   └── sample_submission.csv
   ```
3. `notebooks/01_preprocessing.ipynb`부터 순서대로 실행 (03~09는 `playground-series-s6e7/processed/cv_folds.csv`를 공유하므로 03을 먼저 실행해야 함)

또는 Kaggle API가 설정돼 있다면:
```bash
kaggle competitions download -c playground-series-s6e7 -p playground-series-s6e7
cd playground-series-s6e7 && unzip playground-series-s6e7.zip
```

## 실행 환경

`notebooks/03`~`09`는 LightGBM(+Optuna)이 필요합니다(`08`은 추가로 PyTorch도 필요).
```bash
pip install lightgbm optuna torch pandas scikit-learn numpy matplotlib seaborn
```
