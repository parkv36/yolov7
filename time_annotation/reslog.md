RGB ONLY

clf = RandomForestClassifier(
n_estimators=300,
max_depth=12,
min_samples_leaf=2,
class_weight='balanced',
random_state=42,
max_features='sqrt'
)

precision recall f1-score support

           0       0.00      0.00      0.00        40
           1       0.00      0.00      0.00        40
           2       1.00      0.68      0.81        40
           3       0.50      1.00      0.67        40
           4       0.50      1.00      0.67        40

    accuracy                           0.54       200

macro avg 0.40 0.53 0.43 200
weighted avg 0.40 0.54 0.43 200

clf = XGBClassifier(
objective='multi:softmax',
num_class=5,
eval_metric='mlogloss',
max_depth=6,
n_estimators=300,
)

Classification Report:
precision recall f1-score support

           0       0.00      0.00      0.00        40
           1       0.00      0.00      0.00        40
           2       1.00      0.85      0.92        40
           3       0.50      1.00      0.67        40
           4       0.50      1.00      0.67        40

    accuracy                           0.57       200

macro avg 0.40 0.57 0.45 200
weighted avg 0.40 0.57 0.45 200

IR ONLY

using same xgboost

Classification Report:
precision recall f1-score support

           0       0.16      0.25      0.19        40
           1       0.28      0.57      0.38        40
           2       0.91      0.53      0.67        40
           3       0.16      0.10      0.12        40
           4       0.14      0.03      0.04        40

    accuracy                           0.29       200

macro avg 0.33 0.30 0.28 200
weighted avg 0.33 0.29 0.28 200

using same random forest

Classification Report:
precision recall f1-score support

           0       0.28      0.72      0.41        40
           1       0.26      0.42      0.32        40
           2       1.00      0.20      0.33        40
           3       0.25      0.03      0.05        40
           4       0.00      0.00      0.00        40

    accuracy                           0.28       200

macro avg 0.36 0.27 0.22 200
weighted avg 0.36 0.28 0.22 200

Three label RGB Random Forest:

Classification Report:
precision recall f1-score support

           0       0.99      1.00      0.99        80
           1       0.87      0.99      0.92        80
           2       1.00      0.70      0.82        40

    accuracy                           0.94       200

macro avg 0.95 0.90 0.91 200
weighted avg 0.94 0.94 0.93 200

Three label RGB XGBoost:

Classification Report:
precision recall f1-score support

           0       1.00      1.00      1.00        80
           1       0.81      1.00      0.89        80
           2       1.00      0.53      0.69        40

    accuracy                           0.91       200

macro avg 0.94 0.84 0.86 200
weighted avg 0.92 0.91 0.90 200

3 class IR better but still shitty, won't explore for now
