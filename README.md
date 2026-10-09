# ROGII Wellbore Geology Prediction

Machine learning pipeline for the ROGII Wellbore Geology Prediction Kaggle competition. The goal is to predict **True Vertical Thickness (TVT)** along a horizontally drilled well, which tells you where the drill bit sits within the surrounding rock layers. Accurate TVT helps keep a well inside the target formation while drilling.

## Approach

**1. Exploratory data analysis**
Each well has two files: a horizontal well log (measured depth, trajectory, gamma ray, formation surface depths) and a reference "typewell" log with known TVT. I visualised well trajectories against geological formation surfaces and compared gamma ray (GR) patterns between the two logs.

**2. Matching horizontal wells to the typewell**
The two logs don't align row by row, so I used sliding-window cross-correlation (`scipy.signal.correlate`) on normalised GR values. For each window of the horizontal well, I find the best-matching position in the typewell, take its TVT as `matched_TVT`, and keep the correlation strength as `matched_score`. GR was interpolated only for this matching step, and the original column keeps its gaps.

**3. Feature engineering**
Rolling-window statistics (mean and standard deviation of GR) and trajectory slope features (`Z_slope` and its rolling mean and standard deviation) give the models local context. Missing values are left for the tree models to handle.

**4. Modelling**
- **CatBoost regressor** (GPU-accelerated, early stopping) on tabular features
- **LSTM and GRU** sequence models over 30-step windows, built with a well-level train/validation/test split so windows can't leak across wells
- **Stacked ensemble** combining CatBoost, LSTM and GRU predictions with a RidgeCV meta-model

## Tech stack
Python, pandas, NumPy, SciPy, scikit-learn, CatBoost, TensorFlow/Keras, Matplotlib

## Results
CatBoost on the real well data: MAE 6.34, RMSE 8.48 (hold-out set).
