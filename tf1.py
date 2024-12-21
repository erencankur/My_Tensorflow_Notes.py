import pandas as pd
dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")
#print(dataFrame)
#            Fiyat  BisikletOzellik1  BisikletOzellik2
# 0     807.673876       1749.628226       1749.590668
# 1     959.227520       1748.007826       1751.824206
# 2     718.020033       1750.122967       1747.977026
# 3     945.668885       1749.916440       1750.771646
# 4     955.542968       1750.780519       1750.592430
# ..           ...               ...               ...
# 999  1048.892414       1748.656426       1752.539962

import seaborn as sbn
import matplotlib.pyplot as plt

#sbn.pairplot(dataFrame) #bn: veri çerçevesindeki tüm sütunlar arasındaki çiftler için grafikler çizerek veri ilişkilerini anlamaya yardımcı olur.
#plt.show()

#pn: veriyi test/train olarak ikiye ayirmak
#yn: y = wx + b ,x -> feature (özellik), y -> label (hedef değişken)

from sklearn.model_selection import train_test_split

y = dataFrame["Fiyat"].values #bn: y değişkeni, “Fiyat” sütununun değerlerini bir NumPy dizisi olarak tutar. Bu, modelin tahmin etmeye çalıştığı hedef değişkendir.
x = dataFrame[["BisikletOzellik1", "BisikletOzellik2"]].values #bn: x değişkeni, “BisikletOzellik1” ve “BisikletOzellik2” sütunlarını bir NumPy dizisi olarak tutar. Bunlar modelin tahmin yapmak için kullanacağı özelliklerdir.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=15) #bn: train_test_split() fonksiyonu, veri setini eğitim ve test setlerine böler.
#bn: test_size=0.33 parametresi, test setinin veri kümesinin %33’ü kadar olmasını belirtir. Yani, veri kümesinin %33’ü test için, geri kalan %67’si ise eğitim için kullanılacaktır.
#bn: random_state=15 parametresi, rastgeleliği kontrol eder, bu da aynı kod çalıştırıldığında aynı sonuçların elde edilmesini sağlar.
#gn: x_train ve y_train, modelin eğitileceği giriş ve çıkış verilerini içerir.
#gn: x_test ve y_test, modelin doğrulama sırasında performansını değerlendirmek için kullanılacak olan test verilerini içerir.

print(x_train.shape) # (670, 2) #bn: x_train = Eğitim seti için giriş özellikleri (features).
print(x_test.shape)  # (330, 2) #bn: x_test  = Test seti için giriş özellikleri.
print(y_train.shape) # (670,)   #bn: y_train = Eğitim seti için hedef değişkenler (labels).
print(y_test.shape)  # (330,)   #bn: y_test  = Test seti için hedef değişkenler.

#pn: scaling (ölçeklendirme)

from sklearn.preprocessing import MinMaxScaler #bn: verileri belirli bir aralıkta (genellikle [0, 1]) ölçeklendirmek için kullanılan bir dönüştürücüdür.

scaler = MinMaxScaler() #bn: scaler nesnesi oluşturulur ve verileri ölçeklendirmek için kullanılır. Default olarak MinMaxScaler, verileri 0 ile 1 arasına ölçeklendirir.
scaler.fit(x_train) #bn: x_train verisi üzerinden ölçeklendirme işlemi için gerekli olan minimum ve maksimum değerleri hesaplar.

x_train = scaler.transform(x_train) #bn: x_train verisini Min-Max ölçeğinde dönüştürerek (transform) her bir özelliğin değerini 0 ile 1 arasına çeker.
x_test = scaler.transform(x_test) #bn: benzer şekilde x_test verisini dönüştürür.

print(x_train)
print(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore #bn: tf'nin keras API’sinden Sequential sınıfını içe aktarır. Sequential sınıfı, katmanları sırayla ekleyerek oluşturulan bir sinir ağı modelidir.
from tensorflow.keras.layers import Dense # type: ignore #bn: Keras’tan Dense katmanını içe aktarır. Dense, sinir ağı katmanlarındaki her bir düğümün (nöronun) bir önceki katmandaki tüm düğümlere bağlı olduğu tam bağlantılı (fully connected) bir katman türüdür.

model = Sequential() #bn: yeni bir Sequential model nesnesi oluşturulur. Sequential modeli, basit bir katman yığınıdır; yani katmanları sırayla birbirine ekleyerek modelin yapısını oluşturur.

#rn: Giriş Katmanları:
model.add(Dense(4, activation="relu")) #bn: Bu satırlar, modelimize üç adet Dense (tam bağlantılı) katman ekler. Her biri 5 nörondan oluşur ve “ReLU” (Rectified Linear Unit) aktivasyon fonksiyonunu kullanır.
model.add(Dense(4, activation="relu")) #bn: 4: Bu kısım katmandaki nöron sayısını belirler.
model.add(Dense(4, activation="relu")) #bn: activation="relu": ReLU aktivasyon fonksiyonunu kullanır. ReLU, giriş değeri pozitifse aynı değeri, negatifse sıfır döndürür. Bu aktivasyon fonksiyonu, derin öğrenme modellerinde yaygın olarak kullanılır ve modelin doğrusal olmayan ilişkileri öğrenmesine yardımcı olur.
#gn: Bu kodda üç ardışık Dense katmanı tanımlanmıştır; bu da modelin derinliğini artırır ve giriş verilerindeki karmaşık ilişkileri öğrenme yeteneğini geliştirir.

#rn: Çıkış Katmanı:
model.add(Dense(1)) #bn: Bu satır, modelin çıkış katmanını tanımlar ve bu katman 1 nörondan oluşur. Aktivasyon fonksiyonu belirtilmemiş, bu da varsayılan olarak “linear” (doğrusal) aktivasyon fonksiyonu kullanılacağı anlamına gelir. Bu durumda, çıktı değeri doğrudan girişe bağlıdır. Çoğunlukla regresyon problemlerinde kullanılır.

#rn: Modelin Eğitimi:
model.compile(optimizer = "rmsprop" ,loss = "mse") #bn: Modelin derlenmesi, modelin eğitim sürecini tanımlamak için kullanılır. Derleme sırasında, optimizasyon algoritması ve kayıp fonksiyonu gibi önemli parametreler belirlenir.
#bn: optimizer = "rmsprop": RMSprop (Root Mean Square Propagation) optimizasyon algoritmasını kullanır. Bu algoritma, öğrenme oranını dinamik olarak ayarlayarak hızlı ve etkili bir şekilde eğitim yapmayı sağlar. RMSprop, özellikle derin sinir ağlarının eğitiminde yaygın olarak kullanılır ve genellikle stokastik gradyan iniş (SGD) yerine kullanılır.
#bn: loss = "mse": Kayıp fonksiyonunu Mean Squared Error (MSE, Ortalama Kare Hatası) olarak ayarlar. MSE, sürekli değerleri tahmin eden regresyon problemleri için kullanılan bir kayıp fonksiyonudur. Gerçek değerler ile tahmin edilen değerler arasındaki kare hatalarının ortalamasını hesaplar ve modelin ne kadar iyi öğrendiğini ölçer.

model.fit(x_train, y_train, epochs=250) #bn: Bu satır, modelin eğitim sürecini başlatır.
#bn: x_train: Modelin eğitileceği giriş verileridir (özellikler).
#bn: y_train: Modelin hedef çıktılarıdır (etiketler veya tahmin edilmesi gereken değerler).
#bn: epochs=250: Eğitim sürecinde modelin tüm eğitim verisini 250 kez (epoch) görmesini belirtir. Her epoch’ta model, tüm eğitim seti üzerinde bir ileri ve bir geri geçiş yapar.

loss = model.history.history["loss"] #bn: Bu satır, eğitim sürecindeki her bir epoch için kaydedilen kayıp (loss) değerlerini alır.

sbn.lineplot(x = range(len(loss)), y = loss) #bn: Seaborn kütüphanesi (sbn) kullanarak bir çizgi grafiği oluşturur.
#bn: x = range(len(loss)): X ekseninde, epoch numaralarını temsil eden aralık değerlerini (0’dan len(loss)’e kadar) kullanır.
#bn: y = loss: Y ekseninde, her epoch için hesaplanan kayıp değerlerini kullanır.
plt.show()

trainLoss = model.evaluate(x_train, y_train, verbose=0) #bn: Modelin eğitim verisi (x_train ve y_train) üzerindeki kaybını hesaplar ve trainLoss değişkenine kaydeder.
testLoss = model.evaluate(x_test, y_test, verbose=0) #bn: Modelin test verisi (x_test ve y_test) üzerindeki kaybını hesaplar ve testLoss değişkenine kaydeder. Bu, modelin daha önce görmediği veriler üzerindeki performansını değerlendirmek için kullanılır.
#bn: verbose=0: Çıktının bastırılmamasını sağlar. Eğer verbose=1 veya verbose=2 olarak ayarlanırsa, daha ayrıntılı eğitim çıktıları gösterilir.

print(trainLoss) # 639.62548828125 #bn: Eğitim verisi üzerindeki ortalama kayıp değeri.
print(testLoss) # 623.8090209960938 #bn: Test verisi üzerindeki ortalama kayıp değeri.

testTahminleri = model.predict(x_test) #bn: Bu satır, eğitilmiş modelin test veri seti (x_test) üzerindeki tahminlerini üretir.
#bn: model: Daha önce eğitilmiş olan yapay sinir ağı modelidir. Model, model.fit() yöntemiyle eğitim verisi (x_train ve y_train) üzerinde eğitilmiş durumdadır.
#bn: x_test: Modelin performansını değerlendirmek için kullanılan test veri setidir. Bu veri seti modelin daha önce görmediği verilerdir ve genellikle x_train veri setinin bir kısmı olarak ayrılır.
#bn: predict(): Keras modellerinin tahmin yapmasını sağlayan bir yöntemdir. Girdi olarak verilen veri setine göre model, her bir giriş için bir çıktı üretir. Bu çıktılar modelin tahminleridir.
#bn: testTahminleri: model.predict(x_test) fonksiyonu, x_test veri seti üzerindeki her bir giriş için tahmin edilen değerlerin bir dizisini döndürür. Bu diziyi testTahminleri adlı bir değişkende saklarız.
print(testTahminleri) #bn: testTahminleri değişkenindeki tahmin edilen değerleri ekrana yazdırır. Bu değerler, modelin test veri setindeki her bir örnek için yaptığı tahminlerdir.
#gn: Tahminler genellikle bir dizide veya matris şeklinde döner ve her bir eleman, modelin o örnek için tahmin ettiği değeri temsil eder. Örneğin, eğer model bir regresyon modeliyse, her bir eleman sürekli bir sayısal değer olabilir. Eğer model bir sınıflandırma modeliyse, her bir eleman sınıf olasılıkları olabilir.

#rn: Gerçek Değerlerin DataFrame’e Dönüştürülmesi:
tahminDf = pd.DataFrame(y_test, columns=["Gerçek Y"]) #bn: y_test içindeki gerçek etiketlerden (veya gerçek değerlerden) bir DataFrame oluşturur.
print(tahminDf) #bn: Bu DataFrame, test verisi için gerçek değerleri (çıktıları) içerir.

#rn: Tahmin Değerlerinin Seriye Dönüştürülmesi:
testTahminleri = pd.Series(testTahminleri.reshape (330,)) #bn: testTahminleri dizisi, reshape yöntemi ile tek boyutlu bir diziye dönüştürülür. Burada 330, dizinin toplam uzunluğunu belirtir. Yani bu kod testTahminleri dizisini yeniden şekillendirir ve 330 elemanlı tek bir boyutlu diziye dönüştürür.
#bn: testTahminleri: Modelin x_test verisi üzerinde ürettiği tahminlerin bir dizisidir.
#bn: reshape(330,): Bu kod, tahminlerin dizisinin tek bir sütuna sahip olmasını sağlar.
#bn: pd.Series(testTahminleri.reshape(330,)): testTahminleri dizisini bir Pandas Serisine (pd.Series) dönüştürür. Pandas Serisi, DataFrame’e eklenmek için kullanılır.
print(testTahminleri)

#rn: Gerçek ve Tahmin Değerlerinin Birleştirilmesi:
tahminDf = pd.concat([tahminDf,testTahminleri], axis=1) #bn: İki Pandas veri yapısını (tahminDf ve testTahminleri) birleştirir. axis=1 parametresi, bu birleştirmenin sütun bazında yapılacağını belirtir.
#bn: tahminDf: Test veri setinin gerçek değerlerini içeren DataFrame.
#bn: testTahminleri: Modelin test veri seti üzerindeki tahminlerini içeren Pandas Serisi.
tahminDf.columns = ["Gerçek Y", "Tahmin Y"]
print(tahminDf)

#rn: Gerçek ve Tahmin Değerlerinin Görselleştirilmesi:
sbn.scatterplot(x = "Gerçek Y", y = "Tahmin Y", data = tahminDf) #bn: Seaborn kütüphanesi (sbn) kullanarak bir saçılma grafiği (scatter plot) oluşturur.
#bn: x = "Gerçek Y": X eksenine gerçek değerleri (Gerçek Y sütunu) koyar.
#bn: y = "Tahmin Y": Y eksenine modelin tahmin ettiği değerleri (Tahmin Y sütunu) koyar.
#bn: data = tahminDf: Grafiği çizmek için kullanılan veri kaynağı tahminDf DataFrame’idir.
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
#bn: mean_absolute_error (MAE): Tahmin edilen değerler ile gerçek değerler arasındaki ortalama mutlak hatayı hesaplar. Bu, tahminlerin ne kadar yanıldığını ölçmek için kullanılır ve hata miktarını aynı birimde gösterir.
#bn: mean_squared_error (MSE): Tahmin edilen değerler ile gerçek değerler arasındaki ortalama kare hatayı hesaplar. MSE, hatanın karesini aldığı için büyük hataları daha fazla cezalandırır ve bu nedenle model performansını değerlendirirken daha hassastır.

mean_absolute_error(tahminDf["Gerçek Y"], tahminDf["Tahmin Y"]) #bn: tahminDf DataFrame’indeki gerçek değerler (Gerçek Y) ile modelin tahmin ettiği değerler (Tahmin Y) arasındaki ortalama "mutlak" hatayı (MAE) hesaplar.
mean_squared_error(tahminDf["Gerçek Y"], tahminDf["Tahmin Y"]) #bn: tahminDf DataFrame’indeki gerçek değerler (Gerçek Y) ile modelin tahmin ettiği değerler (Tahmin Y) arasındaki ortalama "kare" hatayı (MSE) hesaplar.

dataFrame.describe() #bn: dataFrame adlı pandas DataFrame’inin özet istatistiklerini görüntüler.

yeniBisikletOzellikleri = [[1760, 1758]] #bn: Modelin tahmin yapacağı yeni bir veri örneğini tanımlar. 
yeniBisikletOzellikleri = scaler.transform(yeniBisikletOzellikleri) #bn: Modelin eğitiminde kullanılan özellik ölçekleme (scaling) işlemini yeni veri üzerinde uygular.
#bn: scaler: Özelliklerin ölçeklenmesi için kullanılan bir ön işleme nesnesidir.
#bn: transform(): scaler’ın transform yöntemi, yeni veriyi aynı ölçekleme işlemiyle dönüştürür. Bu, modelin doğru tahmin yapabilmesi için önemlidir çünkü model, eğitim sırasında belirli bir veri ölçeğinde eğitilmiştir ve yeni verinin aynı ölçekle olması gerekir.
print(model.predict(yeniBisikletOzellikleri)) # [[1705.5579]] #bn: Modelin yeni veri üzerinde tahmin yapmasını sağlar.
#bn: model: Daha önce eğitim yapılmış bir makine öğrenimi modelidir.
#bn: predict(): Modelin yeni verilere dayalı tahminler üretmesini sağlar. Burada yeniBisikletOzellikleri ölçeklendirilmiş veridir ve bu veri modeli tahmin yapması için kullanılır.

from tensorflow.keras.models import load_model # type: ignore
model.save("bisiklet_modeli.keras") #bn: model nesnesini bir dosyaya kaydeder.
sonradanCagirilanModel = load_model("bisiklet_modeli.keras") #bn: load_model() fonksiyonu, kaydedilmiş modeli bisiklet_modeli.keras dosyasından yükler ve sonradanCagirilanModel değişkenine atar. Bu, modelin yapısını ve ağırlıklarını geri yükler, böylece model tahmin yapmak için kullanılabilir.
print(sonradanCagirilanModel.predict(yeniBisikletOzellikleri)) # [[1705.5579]] #bn: sonradanCagirilanModel.predict() metodu, modelin tahmin yapmasını sağlar. yeniBisikletOzellikleri girdisini kullanarak model, bu girdiye dayalı tahmin sonuçlarını üretir.
