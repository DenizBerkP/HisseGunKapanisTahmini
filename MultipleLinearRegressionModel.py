import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

hisse = "THYAO"
verilerSaatlik = yf.download(hisse + ".IS", period="2y", interval = "1h")
verilerSaatlik['Saatlik SMA_50'] = verilerSaatlik.ta.sma(length=50, append=False)
verilerSaatlik['Saatlik RSI_14'] = verilerSaatlik.ta.rsi(length=14, append=False)
verilerSaatlik["Saatlik MFI_14"] = verilerSaatlik.ta.mfi(length=14, append=False)
verilerSaatlik["Saatlik MFI_14"] = verilerSaatlik.ta.mfi(length=14, append=False)

verilerSaatlik["Saatlik High"] = verilerSaatlik["High"]
verilerSaatlik["Saatlik Low"] = verilerSaatlik["Low"]
verilerSaatlik["Saatlik Adj Close"] = verilerSaatlik["Adj Close"]

silinecek_sutunlar = ["High", "Low", "Adj Close", "Volume"]
verilerSaatlik = verilerSaatlik.drop(silinecek_sutunlar, axis=1)

verilerSaatlik["Datetime"] = verilerSaatlik.index
new_indexSaatlik = range(len(verilerSaatlik))
verilerSaatlik.index = new_indexSaatlik
silinecek_saatler = [ "12:30:00", "13:30:00", "14:30:00", "15:30:00", "16:30:00", "17:30:00"]
for saat in silinecek_saatler:
    verilerSaatlik = verilerSaatlik[~verilerSaatlik['Datetime'].apply(lambda x: x.strftime("%H:%M:%S")).str.contains(saat)]

verilerSaatlik['Datetime'] = pd.to_datetime(verilerSaatlik['Datetime'])


filtre = verilerSaatlik['Datetime'].dt.strftime('%H:%M:%S') == '09:30:00'
verilerSaatlik.loc[filtre, 'Ilk 1 Saatlik Degisim %'] = ((verilerSaatlik['Close'] - verilerSaatlik['Open']) / verilerSaatlik['Open']) * 100

start_index = 2  # 3. elemanın indeksi 2'dir
for index, row in verilerSaatlik.iloc[start_index:].iterrows():
    if index > 0 and verilerSaatlik.loc[index, 'Datetime'].strftime('%H:%M:%S') == '10:30:00':
        open_price = verilerSaatlik.loc[index - 1, 'Open']
        close_price = verilerSaatlik.loc[index, 'Close']
        percent_change = ((close_price - open_price) / open_price) * 100
        verilerSaatlik.loc[index, 'Ilk 2 Saatlik Degisim %'] = percent_change

start_index = 3  # 3. elemanın indeksi 2'dir
for index, row in verilerSaatlik.iloc[start_index:].iterrows():
    if index > 0 and verilerSaatlik.loc[index, 'Datetime'].strftime('%H:%M:%S') == '11:30:00':
        open_price = verilerSaatlik.loc[index - 1, 'Open']
        close_price = verilerSaatlik.loc[index, 'Close']
        percent_change = ((close_price - open_price) / open_price) * 100
        verilerSaatlik.loc[index, 'Ilk 3 Saatlik Degisim %'] = percent_change  
        verilerSaatlik.loc[index, "3 Saatlik Close"] = verilerSaatlik.loc[index, 'Close']

verilerSaatlik['Datetime'] = pd.to_datetime(verilerSaatlik['Datetime']).dt.date

verilerSaatlik1 = verilerSaatlik[['Datetime', 'Ilk 1 Saatlik Degisim %']]
verilerSaatlik1 = verilerSaatlik1.dropna()
verilerSaatlik2 = verilerSaatlik[['Datetime', 'Ilk 2 Saatlik Degisim %']]
verilerSaatlik2 = verilerSaatlik2.dropna()
verilerSaatlik3 = verilerSaatlik[['Datetime', 'Ilk 3 Saatlik Degisim %']]
verilerSaatlik3 = verilerSaatlik3.dropna()
verilerSaatlik4 = verilerSaatlik[["Datetime","Saatlik SMA_50","Saatlik RSI_14","Saatlik MFI_14", "Saatlik High", "Saatlik Low", "Saatlik Adj Close"]]
verilerSaatlik4 = verilerSaatlik4.dropna()

verilerSaatlik2 = pd.merge(verilerSaatlik1, verilerSaatlik2, on='Datetime', how='inner')
verilerSaatlik3 = pd.merge(verilerSaatlik2, verilerSaatlik3, on='Datetime', how='inner')
verilerSaatlik = pd.merge(verilerSaatlik3, verilerSaatlik4, on='Datetime', how='inner')

verilerSaatlik['Datetime'] = pd.to_datetime(verilerSaatlik['Datetime'])


grouped = verilerSaatlik.groupby('Datetime')
counts = grouped.size()
repeated_dates = counts[counts > 1].index
verilerSaatlik = verilerSaatlik.drop_duplicates(subset='Datetime', keep='first')

startDate = str(verilerSaatlik["Datetime"].iloc[0])
startDate = startDate[:-9]
endDate = str(verilerSaatlik["Datetime"].iloc[-1])
endDate = endDate[:-9]
verilerGunluk = yf.download(hisse + ".IS", start=startDate, end=endDate, interval = "1d")

verilerGunluk["Günlük Değişim"] = ((verilerGunluk['Close'] - verilerGunluk['Open']) / verilerGunluk['Open']) * 100
verilerGunluk['Önceki Gün Değişim'] = verilerGunluk["Günlük Değişim"].shift(1)
verilerGunluk['Önceki Gün Değişim'].fillna(method='bfill', inplace=True)

verilerGunluk['Önceki Gün ROC 14'] = verilerGunluk.ta.roc(length=14, append=False)
verilerGunluk['Önceki Gün ROC 14'] = verilerGunluk['Önceki Gün ROC 14'].shift(1)
verilerGunluk['Önceki Gün ROC 14'].fillna(method='bfill', inplace=True)

verilerGunluk['Önceki Gün Close'] = verilerGunluk["Close"]
verilerGunluk['Önceki Gün Close'] = verilerGunluk['Önceki Gün Close'].shift(1)
verilerGunluk['Önceki Gün Close'].fillna(method='bfill', inplace=True)

verilerGunluk['Önceki Gün High'] = verilerGunluk["High"]
verilerGunluk['Önceki Gün High'] = verilerGunluk['Önceki Gün High'].shift(1)
verilerGunluk['Önceki Gün High'].fillna(method='bfill', inplace=True)

verilerGunluk['Önceki Gün Low'] = verilerGunluk["Low"]
verilerGunluk['Önceki Gün Low'] = verilerGunluk['Önceki Gün Low'].shift(1)
verilerGunluk['Önceki Gün Low'].fillna(method='bfill', inplace=True)

verilerGunluk['Önceki Gün Adj Close'] = verilerGunluk["Adj Close"]
verilerGunluk['Önceki Gün Adj Close'] = verilerGunluk['Önceki Gün Adj Close'].shift(1)
verilerGunluk['Önceki Gün Adj Close'].fillna(method='bfill', inplace=True)

silinecek_sutunlar = ["High", "Low", "Adj Close", "Volume", "Open", "Close"]
verilerGunluk = verilerGunluk.drop(silinecek_sutunlar, axis=1)

verilerGunluk["Datetime"] = verilerGunluk.index
new_indexGunluk =  range(len(verilerGunluk))
verilerGunluk.index = new_indexGunluk

verilerSaatlik['Datetime'] = pd.to_datetime(verilerSaatlik['Datetime'])
verilerGunluk['Datetime'] = pd.to_datetime(verilerGunluk['Datetime'])

verilerGunluk = verilerGunluk.dropna()
verilerBirlesik = pd.merge(verilerSaatlik, verilerGunluk, left_on='Datetime', right_on='Datetime', how='inner')

import seaborn as sns
import matplotlib.pyplot as plt

corr = verilerBirlesik.corr()[["Günlük Değişim"]]
plt.figure(figsize=(15, 8))  # Grafiği daha büyük yapabilirsiniz
sns.heatmap(corr, annot=True)
plt.show()

X = verilerBirlesik.drop(["Günlük Değişim","Datetime"], axis = True)
y = verilerBirlesik["Günlük Değişim"]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) #Fit = Train Data, Predict = Test Data

y_pred = regressor.predict(X_test).round(1)

hesaplama = pd.DataFrame(np.c_[y_test, y_pred], columns = ["Orijinal Hareket", "Tahmin Verileri"]) #c_ Sütun Birleştirme işlemi yapar
print(hesaplama)

print("Training Accuracy : ", regressor.score(X_train, y_train)) 
print("Testing Accuracy :", regressor.score(X_test, y_test)) 

from sklearn.metrics import r2_score
a = r2_score(y,regressor.predict(X))
b = float(a)
plt.figure(figsize=(10, 5)) 

plt.plot(hesaplama.index.values, hesaplama["Orijinal Hareket"], "g", label="Gerçek Veriler")
plt.plot(hesaplama.index.values, y_pred, "*--", label="Tahmin Verileri")

plt.xlabel("Günler")
plt.ylabel("Günlük Değişim %")
plt.title(f"{hisse} Hisse Tahmin Grafiği | Başarı Oranı: {regressor.score(X_test, y_test)} | Test Score Index: {b}")
plt.legend()
