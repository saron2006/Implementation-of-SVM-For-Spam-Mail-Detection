# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import the required packages.

Step 2: Import the dataset to operate on.

Step 3: Split the dataset.

Step 4: Predict the required output. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SARON XAVIER A
RegisterNumber: 212223230197
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix, classification_report
con = confusion_matrix(y_test,y_pred)
con
cl=classification_report(y_test,y_pred)

```

## Output:
### head
![image](https://github.com/user-attachments/assets/d8c17580-4b7e-4723-b185-1fc798e8cbb6)

### Info
![image](https://github.com/user-attachments/assets/cb1ecf0e-df83-42c2-af20-00fcf1a2b71e)

### isnull
![image](https://github.com/user-attachments/assets/d9d633f1-aad5-428f-93b1-ad5ec8527c98)

### Accuracy
![image](https://github.com/user-attachments/assets/f1107507-245a-4edc-93ed-d61241ac8299)

### Confusion matrix and Classification Report
![image](https://github.com/user-attachments/assets/8230c1cc-32dd-496e-9a31-35de8e22d3b1)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
