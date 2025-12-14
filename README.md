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

# Records
- pipeline.apply_random_forest
   - two_category: large category has more high difference from prediction