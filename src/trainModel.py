import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

df = pd.read_csv("data/ML_ready.csv")

X = df.drop('Credit_Score', axis=1).values
y = df['Credit_Score'].values

# Correct for imbalanced labels with Synthetic Minority Oversampling TEchnique
smote = SMOTE()
X, y = smote.fit_resample(X,y)
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=101)

# Normalise data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data  = scaler.transform(test_data)

print("Training model")
rfc = RandomForestClassifier(n_estimators=700, criterion='entropy', max_features="log2", random_state=101, n_jobs=4, verbose=2)
rfc.fit(train_data,train_labels)
predicted_labels = rfc.predict(test_data)

print(classification_report(test_labels, predicted_labels))
print("Accuracy = ", classification_report(test_labels, predicted_labels, output_dict=True)['accuracy'])
print("f1-score = ", classification_report(test_labels, predicted_labels, output_dict=True)['weighted avg']['f1-score'])

dump(rfc, 'models/randomForestClassifier.joblib', compress=3)
