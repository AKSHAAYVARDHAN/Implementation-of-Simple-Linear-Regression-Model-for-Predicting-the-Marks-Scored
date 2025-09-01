# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries.
#### 2.Set variables for assigning dataset values.
#### .Import linear regression from sklearn.
#### 4.Assign the points for representing in the graph.
#### 5.Predict the regression for marks by using the representation of the graph.
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Akshaay Vardhan S
RegisterNumber: 212224220007
*/
```

```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:

### Head values
<img width="2180" height="230" alt="image" src="https://github.com/user-attachments/assets/184db7ee-303a-4f6e-bd2c-3145437c9cec" />

### Tail Values
<img width="2180" height="230" alt="image" src="https://github.com/user-attachments/assets/e3911e4a-4819-4b4b-b02e-d8925d0e4e93" />

### Compare Dataset
<img width="2180" height="922" alt="image" src="https://github.com/user-attachments/assets/3eb6e7c7-30bc-49f9-93b2-2e33989b21da" />

### Predication values of X and Y
<img width="2180" height="110" alt="image" src="https://github.com/user-attachments/assets/13bba2a6-c84a-49ed-b244-cb8f874f34e5" />

### Training Set
<img width="2180" height="904" alt="image" src="https://github.com/user-attachments/assets/6dbfda30-6f3d-4349-a5f6-be6c7998c896" />

### Testing Set
<img width="2180" height="904" alt="image" src="https://github.com/user-attachments/assets/37ffd41f-dbe0-4368-b31e-f02e16ee1afe" />

### MSE,MAE and RMSE
<img width="2180" height="112" alt="image" src="https://github.com/user-attachments/assets/ab8f534b-021d-404c-bae3-71c04cc56d3e" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
