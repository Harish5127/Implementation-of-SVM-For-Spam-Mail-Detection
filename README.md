# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the spam dataset and handle encoding properly.
2. Display basic information and check for null values.
3. Extract the message text as features (x) and labels (y) for classification.
4. Split the dataset into training and testing sets.
5. Convert the text data into numerical vectors using CountVectorizer.
6. Train an SVM classifier on the transformed training data.
7. Predict on test data and evaluate model accuracy using accuracy_score.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Harish R
RegisterNumber:  212224230085
*/
```
```
import chardet
file=r"C:\Users\admin\Downloads\spam.csv"
with open(file, 'rb') as rawdata:
    result =chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Downloads\spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
x=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
#Countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
cv= CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:

![Screenshot 2025-05-14 094338](https://github.com/user-attachments/assets/3b7c2f17-312c-41b5-94ec-23c8278fab0d)

![Screenshot 2025-05-14 094400](https://github.com/user-attachments/assets/1a690ee7-454e-4618-b9e6-920bcb62a6a9)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
