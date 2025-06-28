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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../data/telco_churn.csv')

df = df.drop('customerID', axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df = df.dropna()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df_encoded = pd.get_dummies(df, drop_first=True).astype('int')

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_resampled, y_resampled)
y_pred = rf.predict(X_test)

print('Train ACC:', rf.score(X_resampled, y_resampled))
print('Test ACC:', rf.score(X_test, y_test))

print(confusion_matrix(y_test, y_pred))
print('==========================================')
print(classification_report(y_test, y_pred))



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