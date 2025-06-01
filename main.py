import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from preprocessing import preprocess # preprocessing.py dosyasından preprocess fonksiyonunu içe aktarır.

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# Veri Setlerinin Yüklenmesi
# Yarışma için sağlanan eğitim ve test veri setleri okunur.
# 'id' sütunları daha sonra tahmin sonuçları için saklanır ve ana DataFrame'lerden çıkarılır.
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

train_id = train["id"].copy()
test_id = test["id"].copy()

train_df = train.drop("id",axis=1)
test_df = test.drop("id", axis=1)



########################
# Keşifçi Veri Analizi (EDA)
########################
# Veri setlerinin genel yapısı, eksik değerler ve istatistiksel özetleri incelenir.
train_df.info()
train_df.isna().sum()
test_df.info()
test_df.isna().sum()

train_df.describe([0.1, 0.25, 0.5, 0.75, 0.90, 0.99]).T


def plot_numerical_distributions(df, numerical_cols):
    """
    Verilen sayısal sütunların histogramlarını ve çarpıklık (skewness) değerlerini alt grafikler halinde gösterir.
    Ayrıca, çarpıklığı belirli bir eşiğin üzerinde olan sütunları döndürür.
    """
    skew_columns = []
    num_cols = len(numerical_cols)
    num_rows = (num_cols + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 6 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.histplot(x=df[col].dropna(), kde=True, bins=50, ax=axes[i])
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

        sk = skew(df[col].dropna())
        print(f"Skewness for {col}: {sk:.2f}")

        if sk > 1 or sk < -1: # Çarpıklık değeri -1 veya 1'den büyükse listeye eklenir.
            skew_columns.append(col)

    if num_cols % 2 != 0:
        fig.delaxes(axes[-1])


    plt.tight_layout()
    plt.show()
    return skew_columns

# Eğitim veri setindeki sayısal sütunlar belirlenir ve dağılımları çizilir.
numerical_col_train = [col for col in train_df.columns if train_df[col].dtype != 'O']
skew_list_train = plot_numerical_distributions(train_df, numerical_col_train)


# Korelasyon Matrisi
# Sayısal özellikler arasındaki ilişkileri görselleştirmek için korelasyon matrisi çizilir.
correlation_matrix = train_df[numerical_col_train].corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()



########################
# Özellik Mühendisliği (Feature Engineering) & Özellik Çıkarımı
########################
# Mevcut özelliklerden yeni anlamlı özellikler türetilir.
# Bu yeni özellikler, modelin öğrenme yeteneğini artırmayı hedefler.

train_df['New_HeartRateMinute'] = train_df['Heart_Rate'] / train_df['Duration'] # Dakika başına kalp atış hızı
train_df['New_HeartRateDuration'] = train_df['Heart_Rate'] * train_df['Duration'] # Kalp atış hızı ve sürenin çarpımı
train_df['New_DurationBodyTemp'] = train_df['Duration'] * train_df['Body_Temp'] # Süre ve vücut sıcaklığının çarpımı
train_df['New_AgeHeartRate'] = train_df['Age'] * train_df['Heart_Rate'] # Yaş ve kalp atış hızının çarpımı
train_df['New_AgeBodyTemp'] = train_df['Age'] * train_df['Body_Temp'] # Yaş ve vücut sıcaklığının çarpımı
train_df['New_BodyArea'] = train_df['Height'] * train_df['Weight'] # Vücut alanı

# Süre özelliğini kategorik aralıklara böler.
train_df['New_DurationCategory'] = pd.qcut(train_df['Duration'], q=4, labels=['Sedentary', 'Lightly_Active', 'Moderately_Active', 'Very_Active'])

# Yaş özelliğini kategorik aralıklara böler.
bins = [20, 30, 45, 60, 70, 80]
labels = ['Young', 'Adult', 'Middle_Aged', 'Old_Age', 'Old']
train_df['New_AgeCategory'] = pd.cut(train_df['Age'], bins=bins, labels=labels, right=False)



########################
# Kodlama (Encoding) & Ölçekleme (Scaling)
########################

def label_encoder(dataframe, binary_col):
    """İkili kategorik sütunları sayısal değerlere dönüştürür (0 ve 1)."""
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """Nominal kategorik sütunları One-Hot Encoding ile dönüştürür.
    drop_first=True ile çoklu doğrusallık sorununu önlemek için ilk kategorik sütun düşürülür.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# 'Sex' sütunu Label Encoding ile dönüştürülür.
label_encoder(train_df, 'Sex')

# Eşsiz değer sayısı 2 ile 10 arasında olan kategorik sütunlar belirlenir ve One-Hot Encoding uygulanır.
categorical_cols = [col for col in train_df.columns if 2 < train_df[col].nunique() < 10]
train_df = one_hot_encoder(train_df,categorical_cols, drop_first=True)


# Sayısal sütunlar (20'den fazla eşsiz değeri olan ve nesne tipi olmayanlar) StandardScaler ile ölçeklenir.
# 'Calories' hedef değişken olduğu için ölçeklemeden çıkarılır.
scaler = StandardScaler()
scale_cols = [col for col in train_df.columns if train_df[col].nunique() > 20 and train_df[col].dtypes != 'O']
scale_cols.remove('Calories')

train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])


########################
# Modelleme
########################

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
import optuna
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# Log RMSE fonksiyonu ve scorer
# log1p dönüşümü, hedef değişkenin çarpıklığını azaltarak modelin performansını artırabilir.
# np.maximum(0, ...) negatif tahminleri önlemek için kullanılır.
def log_rmse_func(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# make_scorer, özel bir skorlama fonksiyonunu scikit-learn'ün çapraz doğrulama fonksiyonlarıyla uyumlu hale getirir.
log_rmse_scorer = make_scorer(log_rmse_func, greater_is_better=False) # False, daha küçük skorun daha iyi olduğunu belirtir (RMSE için).


def compare_models(X, y, random_state=42, cv=3):
    """
    Belirtilen regresyon modellerini çapraz doğrulama ile karşılaştırır ve Log RMSE skorlarını döndürür.
    """
    models = {
        #"Linear Regression": LinearRegression(), # Yorum satırı, ihtiyaca göre dahil edilebilir.
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        #"ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        #"Huber Regressor": HuberRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=random_state),
        "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=random_state),
        "Bagging Regressor": BaggingRegressor(n_estimators=50, random_state=random_state, n_jobs=-1),
        #"KNN": KNeighborsRegressor(),
        #"SVR": SVR(),
        #"Kernel Ridge": KernelRidge(),
        "XGBoost": XGBRegressor(n_estimators=50, random_state=random_state, verbosity=0, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=50, random_state=random_state, verbose=-1, n_jobs=-1),
        "CatBoost": CatBoostRegressor(verbose=False, random_state=random_state)
    }

    results = []
    for name, model in models.items():
        try:
            # cross_val_score negatif skor döndürdüğü için '-' ile pozitif yapılır.
            score = -cross_val_score(model, X, y, scoring=log_rmse_scorer, cv=cv).mean()
            results.append({
                'Model': name,
                'Log RMSE': round(score, 4)
            })
        except Exception as e:
            results.append({
                'Model': name,
                'Log RMSE': f"Error: {e}"
            })

    return pd.DataFrame(results).sort_values("Log RMSE", ascending=True).reset_index(drop=True)

# Özellikler (X) ve hedef değişken (y) ayrılır.
X = train_df.drop(['Calories'], axis=1)
y = train_df["Calories"]

# Veri seti eğitim ve doğrulama setlerine ayrılır.
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)

# Modeller karşılaştırılır ve sonuçlar yazdırılır.
results = compare_models(X_train, y_train)
print(results)

# Yukarıdaki çıktıda gösterilen model sıralaması (örneğin CatBoost en iyi performansı göstermiş).
#                Model  Log RMSE
# 0           CatBoost    0.0628
# 1      Random Forest    0.0644
# 2  Bagging Regressor    0.0644
# 3            XGBoost    0.0698
# 4           LightGBM    0.0842
# 5      Decision Tree    0.0876
# 6  Gradient Boosting    0.1314
# 7   Ridge Regression    0.3898
# 8   Lasso Regression    0.4579
# 9           AdaBoost    0.5241


##############################################
# Model ve Hiperparametre Optimizasyonu
##############################################
# En iyi performansı gösteren CatBoost modeli için hiperparametre optimizasyonu yapılır.
catboost_model = CatBoostRegressor(random_state=42, verbose=False)

def plot_importance(model, features, num=len(X), save=False):
    """Modelin özellik önem düzeylerini görselleştirir."""
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# Optuna ile CatBoost için hiperparametre optimizasyonu
# objective fonksiyonu, Optuna'nın optimize edeceği amacı (Log RMSE'yi minimize etmek) tanımlar.
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 700, 1200),
        "depth": trial.suggest_int("depth", 8, 14),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 100, 300),
        "verbose": 0,
        "loss_function": "RMSE",
        "random_seed": 42
    }

    model = CatBoostRegressor(**params)

    # 5-Fold Çapraz Doğrulama ile model değerlendirilir.
    scores = cross_val_score(model, X_train, y_train,
                             scoring=log_rmse_scorer,
                             cv=5, n_jobs=-1)

    return np.mean(scores) # Minimize edilecek ortalama Log RMSE döndürülür.

# Optuna çalışması başlatılır ve en iyi parametreler bulunur.
study = optuna.create_study(direction="minimize") # RMSE minimize edildiği için yön 'minimize'dir.
study.optimize(objective, n_trials=30) # 30 deneme yapılır.


print("Best Log RMSE:", study.best_value)
print("Best Params:", study.best_params)

# Bulunan en iyi parametreler manuel olarak atanır (veya doğrudan study.best_params kullanılabilir).
best_params = {'iterations': 1190,
               'depth': 12,
               'learning_rate': 0.08111993275911336,
               'l2_leaf_reg': 4.006075660366656,
               'bagging_temperature': 0.5861436328046665,
               'border_count': 109
               }

# Final modeli en iyi parametrelerle eğitilir.
final_model = catboost_model.set_params(**best_params).fit(X_train, y_train)
# Özellik önem düzeyleri çizilir.
plot_importance(final_model, X_train)

# Doğrulama Seti Üzerinde Değerlendirme
y_pred_val = final_model.predict(X_val)

log_rmse = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(y_pred_val)))
print(f"Validation score : {log_rmse}") # Doğrulama seti Log RMSE skoru yazdırılır.

# Test Veri Seti Üzerinde Tahmin ve Gönderi Dosyası Oluşturma
# preprocess fonksiyonu, test veri setine eğitimde uygulanan aynı ön işleme adımlarını uygular.
test_data = preprocess(test_df)
y_pred_test = final_model.predict(test_data)

# Negatif tahminleri 0'a çekerek submission dosyasını oluşturur.
submission = pd.DataFrame({
    'id':test_id,
    'Calories':np.maximum(0, y_pred_test)
})

# Submission dosyası CSV formatında kaydedilir.
submission.to_csv("submission.csv", index=False)