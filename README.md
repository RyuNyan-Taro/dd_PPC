# dd_PPC
Competition scripts for DataDriven

competition: https://www.drivendata.org/competitions/305/competition-worldbank-poverty/page/965/

# Scores
## 36.559
- model: simple RandomForest
- process: preprocess.standardized_with_numbers -> model.fit_random_forest
- It is for the submission test
## 22.148
- model simple RandomForest
- process: pipeline.apply_random_forest
- some category labels are used

## 21.492
- model: simple RandomForest
- process: pipeline.fit_and_predictions
- a few category labels and hot encoded region columns are used

## 23.876 
- model: simple lightgbm.LGBMRegressor
- process: pipeline.fit_and_predictions_lightgbm
- it is the first submission by using lightgbm

## 20.451
- model: simple RandomForest
- process: pipeline.fit_and_prediction_random_forest
- add all category columns with encoding

## 10.228
- model: simple lightgbm.LGBMRegressor
- process: pipeline.fit_and_predictions_lightgbm
- modify to use log transformation

## 10.165
- model: simple lightgbm.LGBMRegressor with coxbox
- process: pipeline.fit_and_predictions_lightgbm
- apply to coxbox

## 76.229
- model: lightgm.LGBMRegressor with coxbox and isotonic
- process: pipeline.fit_and_predictions_lightgbm
- apply isotonic regression. But it is not good. <- It was fixed by f808b27. foggoten inverse coxbox was the case.

# 4.48
- model: stacking of some models
- process: pipeline.fit_and_predictions_pipeline
- apply parameter tuning of some single models

# Records
- pipeline.apply_random_forest
   - two_category: large category has more high difference from prediction
- use coxbox
   - lambda = 0.09 is best score for pipeline.fit_and_test_lightgbm: 10.050838758 -> 9.667269918408259
   - it is not standardized, but the competition score is target a 0.4.
- model ansamble
  - 5.524771423370167 (xgboost) -> 4.996441241862218 (xgboost, lightgbm and catboost)
- hyper parameter tuning
  - 8.582353 (ensemble and k-fold mean) -> 6.990600 (xgboost, lightgbm and catboost)
  - Not used some seeds: some seeds are not good for reducing the score.
- Target encoding
  - It was very worse the score (37.428473)
  - Standardization is not good for the score
- GroupKFold
- Target encoding
  - some models increase training score but finally drop because of a decrease test score
