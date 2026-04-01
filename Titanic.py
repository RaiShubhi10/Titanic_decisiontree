import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

titanic_Df = pd.read_csv("Titanic-Dataset.csv")

# print(titanic_Df.isna().sum())

label = OneHotEncoder
Gender = label.fit_transform(titanic_Df["Sex"])
Embark = label.fit_transform(titanic_Df["Embarked"])

input_col = [ 'Pclass','SibSp', 'Parch',  'Fare', 'Embark', 'Gender_value']
target = "Survived"

inputs = titanic_Df[input_col].astype(str).agg(" ".join, axis=1)
# print(inputs)

vectorized = TfidfVectorizer()

titanic_train, titanic_temp , y_train, y_temp = train_test_split(inputs,titanic_Df[target], train_size=0.7,random_state=42,stratify=titanic_Df[target])
titanic_val, titanic_test , y_val, y_test = train_test_split(titanic_temp,y_temp, train_size=0.5,random_state=42,stratify=y_temp)

X_train = vectorized.fit_transform(titanic_train)
X_val = vectorized.transform(titanic_val)
X_test = vectorized.transform(titanic_test)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

y_val_pred = model.predict(X_val)

print("Confusion matrix : \n", confusion_matrix(y_val,y_val_pred))
print("Accuracy score : \n", accuracy_score(y_val,y_val_pred))