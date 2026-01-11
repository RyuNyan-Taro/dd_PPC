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
- AI next action
1. Robust Feature Engineering (Survey-Level Aggregates)
Poverty is often relative to the local community. Your current model mostly looks at individual households. Creating features that compare a household to its neighbors (same survey_id) can be very powerful.
I have added a create_survey_aggregates function to dd_PPC/preprocess/_preprocess.py. You can use it to generate features like:
•
Mean/Std of household size in the survey: Is this household larger or smaller than average for this area?
•
Ratios: The household's hsize divided by the survey mean hsize.
•
Demographic ratios: Percentage of children in the survey area.
# Example usage in your pipeline
from dd_PPC.preprocess import create_survey_aggregates
survey_features = create_survey_aggregates(train_df)
train_x = pd.concat([train_x, survey_features], axis=1)
2. Move Beyond Manual Categorical Encoding
Your current _category_encoding uses manual mapping. While this captures some order, it might miss complex relationships. Target Encoding is often much more effective for high-cardinality categorical variables.
I've added a target_encode function to dd_PPC/preprocess/_preprocess.py. It uses smoothing to prevent overfitting.
•
Apply this to columns like sector1d, dweltyp, and even region if they aren't already binary.
•
Tip: Be careful to apply target encoding within your cross-validation loops to avoid data leakage.
3. Implement Robust Cross-Validation (GroupKFold)
Currently, you seem to split your data using a single survey_id (300000). This is very risky as that one survey might not be representative. Top competitors use GroupKFold (grouped by survey_id) to ensure that households from the same survey are always either entirely in the training set or entirely in the validation set.
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=X['survey_id']):
    # Train on X.iloc[train_idx], Validate on X.iloc[val_idx]
4. Optimize the Target Transformation
You are using a fixed Box-Cox lambda of 0.09. While this helped you get to 5.0, it might not be optimal for all models or all features.
•
Dynamic Optimization: Use scipy.stats.boxcox without a fixed lambda during training to find the best value for your specific feature set.
•
Log Transformation: Sometimes a simple np.log1p works better when combined with specific features.
•
Quantile Transformer: Try sklearn.preprocessing.QuantileTransformer (output_distribution='normal') to force your target into a normal distribution.
5. Weighted Ensembling and Stacking
Instead of np.mean(pred_vals, axis=0), use a weighted average or a meta-learner (Stacking).
•
Weighted Average: Give more weight to your best performing model (e.g., LightGBM usually performs best on this type of data).
•
Stacking: Use the predictions of XGBoost, LightGBM, and CatBoost as features for a simple Ridge regression or another LightGBM model. This allows the meta-model to learn which base model is more "trustworthy" for different types of households.
6. Hyperparameter Tuning (Optuna)
Your current model parameters are mostly defaults or simple guesses. Use Optuna to tune:
•
learning_rate (try smaller values like 0.005 - 0.01 with more iterations)
•
num_leaves and max_depth
•
feature_fraction and bagging_fraction (crucial for preventing overfitting)
7. Exploit the Competition Metric
The metric is a weighted average of consumption error and poverty rate error.
•
Your use of Isotonic Regression is a great start for calibrating poverty rates.
•
Ensure your Isotonic Regression is fitted on a proper validation set (from GroupKFold), not just the training set, to avoid over-optimistic calibration.
By implementing GroupKFold and Survey-level aggregates first, you should see a significant jump in your local validation score that better reflects your actual leaderboard performance.