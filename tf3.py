import numpy as np
import pandas as pd

dataFrame = pd.read_excel("maliciousornot.xlsx")

import matplotlib.pyplot as plt
import seaborn as sbn

#sbn.countplot(x="Type", data=dataFrame) #bn: "Type” sütunundaki kategorilerin sayısal dağılımını gösteren bir grafiği oluşturur. Bu, her bir “Type” sınıfının veri kümesindeki frekansını görselleştirir.
#plt.show()

dataFrame.corr()["Type"].sort_values() #bn: “Type” sütunu ile veri kümesindeki diğer sayısal sütunlar arasındaki korelasyon katsayılarını hesaplar ve bunları küçükten büyüğe sıralar. Böylece hangi özelliklerin “Type” sütunuyla daha güçlü bir ilişkiye sahip olduğunu anlayabilirsin.

#dataFrame.corr()["Type"].sort_values().plot(kind="bar") #bn: plot(kind="bar"): Sıralanmış korelasyonları bar (çubuk) grafiği olarak görselleştirir. Bu görselleştirme, hangi özelliklerin “Type” sütunuyla pozitif veya negatif bir korelasyona sahip olduğunu hızlıca anlamanı sağlar.
#plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

y = dataFrame["Type"].values
x = dataFrame.drop("Type", axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=15)

scaler = MinMaxScaler ()
scaler.fit(x_train) #bn: Eğitim verisi üzerinde fit işlemi, min ve max değerlerini öğrenir
x_train = scaler.transform(x_train) #bn: Eğitim verisi üzerinde ölçekleme işlemi, veriyi 0-1 aralığına dönüştürür
#gn: üstteki iki satır, şununla aynı işlevi görür: x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) #bn: Test verisi de aynı min-max değerlere göre ölçeklenir

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print(x_train.shape) # (383, 30) #bn: Eğitim veri setinde 383 adet örnek var ve her bir örnek 30 özellikten oluşuyor.

model = Sequential() #bn: Sequential model, Keras’ta kullanılan en basit model türüdür. Katmanlar sırayla (sekans halinde) eklenir.
model.add(Dense(units=30, activation="relu")) #bn: Bu satır, 30 nörondan oluşan ilk gizli katmanı oluşturur.
#bn: Dense: Her nöronun önceki katmandaki tüm nöronlara tam bağlantılı olduğu anlamına gelir.
#bn: units=30: Bu katmanda 30 nöron olduğunu belirtir.
#bn: activation="relu": Aktivasyon fonksiyonu olarak ReLU (Rectified Linear Unit) kullanılır. ReLU, negatif değerleri sıfırlar ve pozitif değerleri olduğu gibi bırakır. Bu fonksiyon, modeli doğrusal olmayan hale getirir, bu da modelin daha karmaşık örüntüleri öğrenmesine yardımcı olur.
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=1, activation="sigmoid")) #bn: Bu satır, çıktı katmanını tanımlar.
#bn: units=1: Tek bir nöron var, çünkü bu bir ikili sınıflandırma problemidir (malicious/not malicious gibi).
#bn: activation="sigmoid": Sigmoid aktivasyon fonksiyonu, çıktıyı 0 ile 1 arasında bir olasılık değeri olarak verir. Bu, ikili sınıflandırma problemleri için uygun bir fonksiyondur.
model.compile(loss="binary_crossentropy", optimizer ="adam")
#bn: loss="binary_crossentropy": İkili sınıflandırma problemlerinde yaygın olarak kullanılan bir kayıp fonksiyonudur. Modelin tahminleri ile gerçek etiketler arasındaki farkı hesaplar ve bu farkı minimize etmeye çalışır.
#bn: optimizer="adam": Adam optimizasyon algoritması kullanılıyor. Bu algoritma, öğrenme oranını dinamik olarak ayarlar ve modelin daha hızlı ve etkili bir şekilde optimize edilmesini sağlar.

model.fit(x=x_train, y=y_train, epochs=700, validation_data=(x_test,y_test), verbose=1) #bn: Bu satır, modeli eğitim verisi üzerinde eğitmeye başlar ve her eğitim adımında ilerlemeyi gösterir.
#bn: model.fit(...): Bu fonksiyon, modeli verilen eğitim verisi üzerinde eğitir. Model, x_train giriş verilerini ve y_train hedef değerlerini kullanarak öğrenir.
#bn: x=x_train: Modelin öğrenmeye çalışacağı giriş verileri (özellikler).
#bn: y=y_train: Modelin öğrenmeye çalışacağı hedef değerler (etiketler).
#bn: epochs=700: Model 700 kez tüm eğitim veri seti üzerinden geçerek eğitilecek. Her bir “epoch”, tüm veri seti üzerinde yapılan bir tam geçişi temsil eder. Daha fazla epoch, modelin daha fazla öğrenme şansı bulduğu anlamına gelir, ancak çok fazla epoch, aşırı öğrenmeye (overfitting) neden olabilir.
#bn: validation_data=(x_test, y_test): Bu parametre, modelin her epoch sonunda eğitimden bağımsız olarak doğrulama verisi (x_test ve y_test) ile performansını değerlendirmesini sağlar. Doğrulama verisi modelin eğitim sırasında gördüğü verilerden farklıdır, bu da modelin genelleme yeteneğini ölçmeye yardımcı olur.
#bn: verbose=1: Eğitim sırasında konsola detaylı çıktı verir. verbose=1, her epoch için kayıp değeri ve doğrulama sonuçlarını ekranda görmenizi sağlar. Eğer verbose=0 olsaydı, eğitim süreci sırasında hiçbir bilgi ekrana yazdırılmazdı.

modelKaybi = pd.DataFrame(model.history.history)
#bn: Modelin her bir epoch sonunda eğitim ve doğrulama seti üzerindeki kayıp ve diğer metrikleri (örneğin, doğruluk) içerir.
#bn: model.history.history’deki veriler bir DataFrame formatına dönüştürülür. Bu sayede bu veriler üzerinde daha kolay işlemler yapılabilir, örneğin görselleştirme.
modelKaybi.plot() #bn: ğitim sürecinde elde edilen kayıp ve doğrulama kaybı (ve varsa diğer metrikler) çubuk ya da çizgi grafiği olarak çizilir. Varsayılan olarak, plot() fonksiyonu çizgi grafiği çizer.
plt.show()
#gn: Bu kodlar, modelin eğitim süreci sırasında elde edilen kayıp (ve varsa diğer metrikler) değerlerini görselleştirmek için kullanılır.

#pn: overfitting:
model = Sequential()

model.add(Dense(units=30, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer ="adam")

earlyStopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25) #bn: Modelin aşırı öğrenmesini (overfitting) engellemek için EarlyStopping yöntemi kullanılıyor.
#bn: monitor="val_loss": Modelin doğrulama setindeki kayıp fonksiyonu (validation loss) izlenir. Bu, modelin eğitim verilerine değil, hiç görmediği test verilerine ne kadar iyi genelleme yaptığını gösterir.
#bn: mode="min": val_loss değerinin en düşük olduğu noktayı bulmaya çalışır. Yani, doğrulama kaybı azaldıkça model eğitimi devam eder; doğrulama kaybı artmaya başlarsa modelin daha fazla öğrenmesine gerek olmadığı anlaşılır.
#bn: patience=25: Eğer doğrulama kaybı 25 ardışık epoch boyunca iyileşme göstermezse (yani en düşük değerini bulduktan sonra 25 epoch boyunca aynı veya daha kötü bir değer elde edilirse), eğitim durdurulur.
#bn: verbose=1: Eğitim sürecinde EarlyStopping’in ne zaman devreye girdiğini konsola yazdırır.

model.fit(x=x_train, y=y_train, epochs=700, validation_data=(x_test,y_test), verbose=1, callbacks=[earlyStopping])
#bn: epochs=700: Model en fazla 700 epoch boyunca eğitilecek. Ancak EarlyStopping devreye girerse, eğitim erken durdurulabilir.
#bn: validation_data=(x_test, y_test): Eğitim sırasında her epoch’ta modelin test verileri üzerindeki performansı hesaplanır. Bu sayede modelin genelleme kapasitesi ölçülür.
#bn: callbacks=[earlyStopping]: EarlyStopping, bir callback fonksiyonu olarak model eğitim sürecine dahil edilmiştir. Bu callback, eğitim sırasında belirtilen koşulları izler ve erken durdurma kararını verir.

#rn: Amaç: Modelin doğrulama seti üzerindeki performansını iyileştirirken, aşırı öğrenme yapmadan en uygun noktada eğitimi durdurmak.
#rn: Nasıl çalışır: val_loss en düşük seviyeye ulaştığında ve 25 epoch boyunca daha fazla iyileşme olmadığında eğitim durur, bu da modelin daha iyi genelleme yapmasını sağlar.

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

#pn: dropout
model = Sequential()

model.add(Dense(units=30, activation="relu"))
model.add(Dropout(0.6)) #bn: Aşırı öğrenmeyi (overfitting) önlemek için Dropout katmanı eklenir. Bu katman, her eğitim örneğinde %60’lık bir oranla rastgele nöronları “kapatır”, böylece modelin daha genelleştirilmiş öğrenmesini sağlar.

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

tahminlerimiz = (model.predict(x_test) > 0.5).astype("int32")  # 0.5 eşik değerini kullanarak ikili sınıflandırma #bn: predict_classes: Test verileri üzerinde tahminler yapar ve sınıf etiketlerini döndürür.
print(tahminlerimiz)
    # [[0]
    #  [1]
    #  [0]
    #  [1]
    #  [0]
    #  [1]
    #  [0]
    #  ...]

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, tahminlerimiz)) #bn: classification_report: Modelin sınıflandırma performansını değerlendiren bir rapor oluşturur (doğruluk, F1 skoru, vb. ile ilgili bilgiler içerir).
    #               precision    recall  f1-score   support
    # 
    #            0       0.89      0.93      0.91        91
    #            1       0.91      0.85      0.88        74
    # 
    #     accuracy                           0.90       165
    #    macro avg       0.90      0.89      0.90       165
    # weighted avg       0.90      0.90      0.90       165

print(confusion_matrix(y_test, tahminlerimiz)) #bn: confusion_matrix: Gerçek ve tahmin edilen değerler arasındaki ilişkiyi gösteren bir matris oluşturur. Bu matris, modelin hangi sınıfları doğru veya yanlış tahmin ettiğini görmeyi sağlar.
    # [[85  6]
    #  [11 63]]
