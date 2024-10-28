import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

dataFrame = pd.read_excel("maliciousornot.xlsx")

sbn.countplot(x="Type", data=dataFrame)
plt.show()

dataFrame.corr()["Type"].sort_values()

dataFrame.corr()["Type"].sort_values().plot(kind="bar")
plt.show()

y = dataFrame["Type"].values
x = dataFrame.drop("Type", axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=15)

scaler = MinMaxScaler ()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)

model = Sequential()

model.add(Dense(units=30, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer ="adam")

model.fit(x=x_train, y=y_train, epochs=700, validation_data=(x_test,y_test), verbose=1)

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

model = Sequential()

model.add(Dense(units=30, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer ="adam")

earlyStopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

model.fit(x=x_train, y=y_train, epochs=700, validation_data=(x_test,y_test), verbose=1, callbacks=[earlyStopping])

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

model = Sequential()

model.add(Dense(units=30, activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=15, activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=15, activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer ="adam")

earlyStopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)

model.fit(x=x_train, y=y_train, epochs=700, validation_data=(x_test,y_test), verbose=1, callbacks=[earlyStopping])

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

tahminlerimiz = (model.predict(x_test) > 0.5).astype("int32")
print(tahminlerimiz)

print(classification_report(y_test, tahminlerimiz))
print(confusion_matrix(y_test, tahminlerimiz))
