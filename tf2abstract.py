import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sbn

dataFrame = pd.read_excel("merc.xlsx")

print(dataFrame.isnull().sum())

sbn.histplot(dataFrame["price"])
plt.show()
sbn.countplot(dataFrame["year"])
plt.show()

dataFrame = dataFrame.select_dtypes(include=['float64', 'int64'])
dataFrame = dataFrame.drop("transmission", axis=1)
dataFrame.corr()
dataFrame.corr()["price"].sort_values()

sbn.scatterplot(x="mileage", y="price", data=dataFrame)
plt.show()

dataFrame.sort_values("price", ascending=True).head(20)
dataFrame.sort_values("price", ascending=False).head(20)

yuzdeDoksanDokuzDf = dataFrame.sort_values("price", ascending=False).iloc[131:]
plt.figure(figsize=(7,5))
sbn.histplot(yuzdeDoksanDokuzDf["price"])
plt.show()

dataFrame.groupby("year").mean()["price"]

yuzdeDoksanDokuzDf[yuzdeDoksanDokuzDf.year != 1970].groupby("year").mean()["price"]

dataFrame = yuzdeDoksanDokuzDf
dataFrame = dataFrame[dataFrame.year != 1970]
dataFrame.groupby("year").mean()["price"]

y = dataFrame["price"].values
x = dataFrame.drop("price",axis=1).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test) ,batch_size=250, epochs=300)

kayipVerisi = pd.DataFrame(model.history.history)
kayipVerisi.plot()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
tahminDizisi = model.predict(x_test)

mean_absolute_error(y_test,tahminDizisi)

plt.scatter(y_test,tahminDizisi)
plt.show()

dataFrame.iloc[2]
yeniArabaSeries = dataFrame.drop("price",axis=1).iloc[2]
yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshape(-1,5))
print(model.predict(yeniArabaSeries)) # [[61581.69]]
