import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# 1. VERİ YÜKLEME
df = pd.read_csv('train.csv')

# 2. EKSİK DEĞERLERİ İSTATİSTİKSEL OLARAK DOLDURMA (IMPUTATION)
# Kategorik veriler: NaN değeri "bu özellik yok" anlamına gelenler
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
             'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
for col in none_cols:
    df[col] = df[col].fillna('None')

# Sayısal veriler: NaN değeri 0 anlamına gelenler (Örn: Garajı olmayan evin garaj alanı 0'dır)
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
             'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    df[col] = df[col].fillna(0)

# LotFrontage: Arsa cephesini mahalle bazlı medyan ile dolduruyoruz (Hassas İstatistik)
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# Diğer kategorikler için En Çok Tekrar Eden (Mode) değerle doldurma
mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for col in mode_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. AYKIRI DEĞER (OUTLIER) TEMİZLİĞİ
# GrLivArea > 4000 ve SalePrice < 300000 olan sapan noktaları veri setinden çıkarıyoruz.
df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)

# 4. HEDEF DEĞİŞKEN NORMALLEŞTİRME (LOG TRANSFORMATION)
# Dağılımı normalleştirmek ve varyansı dengelemek için.
df["SalePrice"] = np.log1p(df["SalePrice"])

# 5. KORELASYON ANALİZİ VE ISI HARİTASI
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
top_corr_features = numeric_df.corr()['SalePrice'].sort_values(ascending=False).head(10).index
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Konut Fiyatları ile En İlişkili 10 Değişken")
plt.show()

# 6. ÇOKLU DOĞRUSAL REGRESYON MODELİ (OLS)
# En güçlü açıklayıcı 5 değişkeni seçiyoruz.
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']
X = df[features]
y = df['SalePrice']

# Sabit terim (intercept) ekleme ve Veri Bölme
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# OLS Modelini Kurma ve Eğitme
model = sm.OLS(y_train, X_train).fit()

# 7. SONUÇLARI YAZDIRMA VE TAHMİN ANALİZİ
print(model.summary())

# Tahmin vs Gerçek Değerler Grafiği
y_pred = model.predict(X_test)
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Gerçek Fiyatlar (Log)')
plt.ylabel('Tahmin Edilen Fiyatlar (Log)')
plt.title('Tahmin Başarısı: Gerçek vs Tahmin')
plt.show()

print("\nVeri temizleme, görselleştirme ve modelleme başarıyla tamamlandı!")