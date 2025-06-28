# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix

# df = pd.read_csv('../data/telco_churn.csv')

# df = df.drop('customerID', axis=1)

# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# df = df.dropna()

# df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# df_encoded = pd.get_dummies(df, drop_first=True).astype('int')

# X = df_encoded.drop('Churn', axis=1)
# y = df_encoded['Churn']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# rf = RandomForestClassifier(
#     n_estimators=100,
#     class_weight='balanced',
#     random_state=42
# )
# rf.fit(X_train, y_train)
# print('Train ACC:', rf.score(X_train, y_train))
# print('Test ACC:', rf.score(X_test, y_test))



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE

# df = pd.read_csv('../data/telco_churn.csv')

# df = df.drop('customerID', axis=1)

# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# df = df.dropna()

# df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# df_encoded = pd.get_dummies(df, drop_first=True).astype('int')

# X = df_encoded.drop('Churn', axis=1)
# y = df_encoded['Churn']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# rf = XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=5,
#     use_label_encoder=False,
#     eval_metrics="logloss",
#     random_state=42
# )
# rf.fit(X_resampled, y_resampled)
# y_pred = rf.predict(X_test)

# print('Train ACC:', rf.score(X_resampled, y_resampled))
# print('Test ACC:', rf.score(X_test, y_test))

# print(confusion_matrix(y_test, y_pred))
# print('==========================================')
# print(classification_report(y_test, y_pred))






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, precision_recall_curve
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt


# Load and prepare data
df = pd.read_csv('../data/telco_churn.csv')
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in X.columns if col not in num_cols] + ['SeniorCitizen']

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# SMOTETomek and Optuna
def objective(trail):
    n_estimators = trail.suggest_int("n_estimators", 50, 300)
    max_depth = trail.suggest_int("max_depth" ,3, 20)
    min_samples_split = trail.suggest_int("min_samples_split", 2, 10)
    class_weight = trail.suggest_categorical("class_weight", [None, 'balanced', 'balanced_subsample'])

    # Full pipeline
    clf_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=42
        ))
    ])

    # Apply preprocessing and resampling
    X_train_transformed = preprocessor.fit_transform(X_train)
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X_train_transformed, y_train)

    # Fit and evaluate
    clf_pipeline.named_steps['clf'].fit(X_resampled, y_resampled)
    y_pred = clf_pipeline.named_steps['clf'].predict(preprocessor.transform(X_test))

    return recall_score(y_test, y_pred, pos_label=1)    # It gives recall score for class 1 (churn)

# Optimization
study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=50)

# Train final model with best params
best_params = study.best_params
final_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('clf', RandomForestClassifier(
        **best_params,
        random_state=42
    ))
])

X_train_transformed = final_pipeline.named_steps['preprocessing'].fit_transform(X_train)
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train_transformed, y_train)
final_pipeline.named_steps['clf'].fit(X_resampled, y_resampled)

# Evaluate the test set
X_test_transformed = preprocessor.transform(X_test)
y_proba = final_pipeline.named_steps['clf'].predict_proba(X_test_transformed)[:, 1]
y_test_pred = (y_proba > 0.55).astype(int)

print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))

# Precision-Recall Curve Plot
# precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, precision[:-1], label='Precision', color='blue')
# plt.plot(thresholds, recall[:-1], label='Recall', color='green')
# plt.xlabel("Threshold")
# plt.ylabel("Score")
# plt.title("Precision-Recall vs Threshold")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

joblib.dump(final_pipeline, '../api/model.pkl')