import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sbn

dataFrame = pd.read_excel("merc.xlsx")

#print(dataFrame.isnull().sum()) #bn: Veri kümesindeki her sütun için eksik verilerin sayısını gösterir.

#sbn.histplot(dataFrame["price"]) #bn: "price" sütunundaki verilerin nasıl dağıldığını gösteren bir histogram grafiği oluşturur.
#plt.show()
#sbn.countplot(dataFrame["year"]) #bn: "year" sütunundaki her benzersiz yıl için bir sayı grafiği oluşturur.
#plt.show()

dataFrame = dataFrame.select_dtypes(include=['float64', 'int64']) #bn: Sadece sayısal sütunlar seçilir.
#dataFrame = dataFrame.drop("transmission", axis=1) #bn: Ya da bunun yerine istenilen sütun çıkarılır.
dataFrame.corr() #bn: corr() fonksiyonu, bir pandas DataFrame’inde yer alan sayısal sütunlar arasındaki korelasyon (ilişki) katsayılarını hesaplar ve bir korelasyon matrisi döndürür.
    #                 year     price   mileage       tax       mpg  engineSize
    # year        1.000000  0.520712 -0.738027  0.012480 -0.094626   -0.142147
    # price       0.520712  1.000000 -0.537214  0.268717 -0.438445    0.516126
    # mileage    -0.738027 -0.537214  1.000000 -0.160223  0.202850    0.063652
    # tax         0.012480  0.268717 -0.160223  1.000000 -0.513742    0.338341
    # mpg        -0.094626 -0.438445  0.202850 -0.513742  1.000000   -0.339862
    # engineSize -0.142147  0.516126  0.063652  0.338341 -0.339862    1.000000
dataFrame.corr()["price"].sort_values() #bn: Oluşturulan korelasyon matrisinden price sütunu ile diğer sütunlar arasındaki korelasyon katsayılarını seçer ve küçükten büyüğe sıralar.
    # mileage      -0.537214
    # mpg          -0.438445
    # tax           0.268717
    # engineSize    0.516126
    # year          0.520712
    # price         1.000000
    # Name: price, dtype: float64

#sbn.scatterplot(x="mileage", y="price", data=dataFrame) #bn: Dağılım grafiği (scatter plot), iki değişken arasındaki ilişkiyi görselleştirmek için kullanılır.
#plt.show()

dataFrame.sort_values("price", ascending=True).head(20) #bn: price sütununa göre verileri küçükten büyüğe sıralar (ascending order).
dataFrame.sort_values("price", ascending=False).head(20) #bn: price sütununa göre verileri büyükten küçüğe sıralar (descending order).

yuzdeDoksanDokuzDf = dataFrame.sort_values("price", ascending=False).iloc[131:]
#plt.figure(figsize=(7,5))
#sbn.histplot(yuzdeDoksanDokuzDf["price"])
#plt.show()

dataFrame.groupby("year").mean()["price"] #bn: Veri çerçevesindeki year sütununa göre verileri gruplar ve her grup için price sütununun ortalamasını hesaplar.
    # year
    # 1970    24999.000000
    # 1997     9995.000000
    # 1998     8605.000000
    #  ...             ...
    # 2020    35433.282337
    # Name: price, dtype: float64

yuzdeDoksanDokuzDf[yuzdeDoksanDokuzDf.year != 1970].groupby("year").mean()["price"] #bn: year = 1970 olanları çıkarttı.

dataFrame = yuzdeDoksanDokuzDf
dataFrame = dataFrame[dataFrame.year != 1970]
dataFrame.groupby("year").mean()["price"]

y = dataFrame["price"].values #bn: .values, bu sütunu bir NumPy array formatına çevirir.
#bn: y, price sütununun değerlerinden oluşan bir NumPy dizisidir ve tahmin edilecek değişkeni (fiyat) temsil eder.
x = dataFrame.drop("price",axis=1).values #bn: Veri çerçevesinden “price” sütununu çıkarır. axis=1 ifadesi, satır yerine sütunun çıkarılacağını belirtir. Bu işlemin amacı, bağımsız değişkenleri (fiyat dışındaki tüm özellikler) almak.
#bn: x, price sütunu dışındaki tüm özellikleri içeren bir NumPy dizisidir. Bu dizide, bağımsız değişkenler yer alır. Bu bağımsız değişkenler, modelin tahmin yapmak için kullanacağı verileri temsil eder (örneğin, bisikletin özellikleri).
#rn: y: Fiyatları temsil eder (bağımlı değişken).
#rn: x: Fiyat dışındaki tüm özellikleri temsil eder (bağımsız değişkenler).

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
#bn: fit_transform: şu iki işlemi birleştirir:
    #bn: fit (uydurma): x_train üzerindeki minimum ve maksimum değerleri hesaplar. Bu değerler, veriyi belirli bir aralığa (varsayılan olarak 0 ile 1) ölçeklendirmek için kullanılır.
    #bn: transform (dönüştürme): x_train’deki verileri bu minimum ve maksimum değerlere göre ölçekler. Tüm değerler, belirlenen aralığa (genellikle 0 ile 1) getirilir.
#rn: Bu aşamada sadece eğitim verisi kullanılarak ölçekleme parametreleri (min ve max değerleri) hesaplanır. Bu işlemde x_train verisi dönüştürülmüş olarak kaydedilir.
x_test = scaler.transform(x_test)
#bn: transform: Bu fonksiyon, daha önce fit_transform ile hesaplanan minimum ve maksimum değerlere göre test verisini ölçekler. Yani fit işlemi sadece eğitim seti ile yapılmışken, test seti de aynı parametrelerle dönüştürülür.
#rn: Bu önemli bir adımdır, çünkü modelin test setine hiç “görmemiş” olduğu verilere dayanarak tahmin yapması gerekir. Bu yüzden test seti için ayrı bir fit işlemi yapmayız; böylece modelin öğrenme sürecine test verisi dahil olmaz.

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

model = Sequential() #bn: Bu, Keras’ta bir model oluşturmak için kullanılan bir yapıdır. Sequential model demek, katmanların sırayla eklendiği bir model anlamına gelir.

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse") #bn: model.compile(): Modelin eğitim aşamasında kullanılacak olan ayarları belirler. İki ana bileşen içerir:
#bn: optimizer=“adam”: Adam optimizasyon algoritması, yaygın olarak kullanılan bir gradyan inişi optimizasyon yöntemidir. Öğrenme oranını dinamik olarak ayarlayarak, modeli hızlı ve etkili bir şekilde optimize etmeye çalışır.
#bn: loss=“mse” (Mean Squared Error - Ortalama Kare Hatası): MSE, regresyon problemlerinde yaygın olarak kullanılan bir kayıp fonksiyonudur. Tahmin edilen değerlerle gerçek değerler arasındaki farkın karesinin ortalamasını alır. Bu fonksiyon, tahminlerin doğruluğunu ölçer ve modelin, hatayı minimize etmesine yardımcı olur. MSE ne kadar küçükse, modelin performansı o kadar iyidir.

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test) ,batch_size=250, epochs=300)
#bn: validation_data=(x_test, y_test): Modeli eğitim sırasında değerlendirmek için kullanılan doğrulama verisi (validation data). Bu veriler, eğitim sırasında modelin doğruluğunu ölçmek için kullanılır. x_test ve y_test, modelin performansını izlemek için eğitim sürecinde kullanılmaz, ama doğrulama amaçlı değerlendirilir.
#bn: batch_size=250: Eğitim sırasında modelin her bir adımda ne kadar veri kullanacağını belirtir. Batch size burada 250 olarak ayarlandığı için, model her bir adımda 250 örnek üzerinde öğrenme yapar. Bu, özellikle büyük veri kümelerinde hafıza kullanımını azaltır ve eğitim sürecini hızlandırabilir.
#bn: epochs=300: Epochs, modelin tüm eğitim verisi üzerinden kaç kez geçeceğini belirtir. Burada model, tüm eğitim setini 300 kez işleyecek.

kayipVerisi = pd.DataFrame(model.history.history) #bn: Bu kod, modelin eğitim sürecindeki kayıp (loss) ve varsa doğrulama (validation) performansını içeren veriyi bir pandas DataFrame formatına dönüştürür.
#kayipVerisi.plot()
#plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
tahminDizisi = model.predict(x_test) #bn: model.predict(x_test): Eğitim sırasında öğrendiği parametreleri kullanarak, x_test verileri için tahminlerde bulunur.

mean_absolute_error(y_test,tahminDizisi) # 3168.4095169862235

#plt.scatter(y_test,tahminDizisi)
#plt.show()

dataFrame.iloc[2] #bn: Bu satır, dataFrame veri çerçevesinin üçüncü satırını (iloc[2] sıfır tabanlı indekse göre) seçer ve içerdiği tüm özellikleri gösterir.
    # year           2020.0
    # price         65980.0
    # mileage        3999.0
    # tax             145.0
    # mpg              28.0
    # engineSize        4.0
    # Name: 3191, dtype: float64
yeniArabaSeries = dataFrame.drop("price",axis=1).iloc[2]
#bn: Burada dataFrame’den “price” sütunu çıkarılıyor, çünkü fiyat sütunu modelin hedef (tahmin edilecek) değişkeni ve bu sütun modelin girdi olarak alacağı veriler arasında yer almamalı.
#bn: Sonrasında yine üçüncü satır seçiliyor (iloc[2]), yani price sütunu hariç diğer sütunlardaki değerler (year, mileage, tax, mpg, engineSize) alınır.
#bn: Sonuç olarak seçilen arabanın fiyat dışındaki özellikleri bir pandas.Series objesi olarak elde edilir.
yeniArabaSeries = scaler.transform(yeniArabaSeries.values.reshape(-1,5))
#bn: Ölçeklendirme: Modeli eğitirken kullanılan veriler MinMaxScaler ile ölçeklendirilmişti. Bu adımda, yeni araba verileri de aynı ölçekleme işleminden geçiriliyor. scaler.transform() fonksiyonu, eğitilirken kullanılan min-max değerlerine göre bu yeni veriyi 0-1 aralığına dönüştürür.
#bn: reshape(-1, 5): yeniArabaSeries’i uygun formatta (5 sütunlu, tek satırlı) bir NumPy dizisi haline getirir, çünkü modelin girdi olarak 2D veri yapısına ihtiyacı var (satır × özellikler).
#bn: 5 sütun: (year, mileage, tax, mpg, engineSize) verileri.
print(model.predict(yeniArabaSeries)) # [[61581.69]]
#bn: Bu adım, eğitimli modele ölçeklenmiş yeni arabanın verilerini verir ve model bu araba için bir fiyat tahmini yapar.
#gn: Modelin tahmin ettiği fiyat: 61,581.69.
#gn: Gerçek fiyat: 65,980.
