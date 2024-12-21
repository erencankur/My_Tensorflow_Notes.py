import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")
print(dataFrame)

sbn.pairplot(dataFrame)
plt.show()

y = dataFrame["Fiyat"].values
x = dataFrame[["BisikletOzellik1", "BisikletOzellik2"]].values 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=15)

print(x_train.shape) # (670, 2)
print(x_test.shape)  # (330, 2)
print(y_train.shape) # (670,)  
print(y_test.shape)  # (330,)  

scaler = MinMaxScaler()
scaler.fit(x_train) 

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(4, activation="relu")) 
model.add(Dense(4, activation="relu")) 
model.add(Dense(4, activation="relu")) 

model.add(Dense(1))

model.compile(optimizer = "rmsprop" ,loss = "mse")

model.fit(x_train, y_train, epochs=250)

loss = model.history.history["loss"]

sbn.lineplot(x = range(len(loss)), y = loss)

plt.show()

trainLoss = model.evaluate(x_train, y_train, verbose=0)
testLoss = model.evaluate(x_test, y_test, verbose=0) 

print(trainLoss)
print(testLoss)

testTahminleri = model.predict(x_test)

print(testTahminleri) 

tahminDf = pd.DataFrame(y_test, columns=["Gerçek Y"])
print(tahminDf)

testTahminleri = pd.Series(testTahminleri.reshape (330,))
print(testTahminleri)

tahminDf = pd.concat([tahminDf,testTahminleri], axis=1) 

tahminDf.columns = ["Gerçek Y", "Tahmin Y"]
print(tahminDf)

sbn.scatterplot(x = "Gerçek Y", y = "Tahmin Y", data = tahminDf) 

plt.show()

mean_absolute_error(tahminDf["Gerçek Y"], tahminDf["Tahmin Y"]) 
mean_squared_error(tahminDf["Gerçek Y"], tahminDf["Tahmin Y"]) 

dataFrame.describe()

yeniBisikletOzellikleri = [[1760, 1758]]
yeniBisikletOzellikleri = scaler.transform(yeniBisikletOzellikleri)
print(model.predict(yeniBisikletOzellikleri))

model.save("bisiklet_modeli.keras")
sonradanCagirilanModel = load_model("bisiklet_modeli.keras")
print(sonradanCagirilanModel.predict(yeniBisikletOzellikleri))
