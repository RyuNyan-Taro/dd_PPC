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

# Records
- pipeline.apply_random_forest
   - two_category: large category has more high difference from prediction