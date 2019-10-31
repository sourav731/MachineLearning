import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Datasets/Iris.csv')

df.head()

df.columns

df.describe()

df.info()

plt.figure(figsize=(20,20))

df['PetalLengthCm'].hist()

df['PetalWidthCm'].hist()

flower1 = df[df['Species']=='Iris-setosa']
flower2 = df[df['Species']=='Iris-versicolor']
flower3 = df[df['Species']=='Iris-virginica']

def plotFig(columnName):
	plt.figure(figsize=(20,20))
	flower1[columnName].plot(label='setosa')
	flower2[columnName].plot(label='virginica')
	flower3[columnName].plot(label='versicolor')

	plt.legend()


plotFig("SepalLengthCm")
plotFig("SepalWidthCm")
plotFig("PetalLengthCm")
plotFig("PetalWidthCm")

mapFlower = {'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}

df['Species'] = df['Species'].map(mapFlower)

df

df.corr()

from sklearn.utils import shuffle
df = shuffle(df)

df.drop(columns=['Id'],inplace=True)

df.corr()

from sklearn.model_selection import train_test_split
X = df.drop(['Species'],axis = 1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))







