import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

train_data = pd.read_csv("lab_ds102/lab2/data/train_worldbank.csv")
test_data = pd.read_csv("lab_ds102/lab2/data/test_worldbank.csv")

features = [
    'gdp_binh_quan_dau_nguoi', 'tong_gdp', 'ty_le_lam_phat', 'ty_le_that_nghiep', 'ty_le_thuong_mai_tren_gdp', 'ty_le_fdi_tren_gdp',
    'dan_so', 'ty_le_dan_so_do_thi', 'tuoi_tho_trung_binh', 'ty_le_tu_vong_tre_so_sinh', 'ty_le_nguoi_dung_internet', 'ty_le_tiep_can_dien',
    'ty_le_nhap_hoc_trung_hoc', 'luong_phat_thai_co2_binh_quan_dau_nguoi'
]

train_X = train_data[features]
test_X = test_data[features]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_X)
X_test = scaler.transform(test_X)

# LOGISTIC REGRESSION
y_train = train_data['label_a1']
y = test_data['label_a1']

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

y_hat = logistic_model.predict(X_test)

print('\nLOGISTIC REGRESSION')
print('Precision:', precision_score(y, y_hat))
print('Recall   :', recall_score(y, y_hat))
print('F1-score :', f1_score(y, y_hat))

# SOFTMAX REGRESSION
y_train = train_data['label_a2']
y = test_data['label_a2']

softmax_model = LogisticRegression(max_iter=1000)
softmax_model.fit(X_train, y_train)

y_hat = softmax_model.predict(X_test)

print('\nSOFTMAX REGRESSION')
print('Precision:', precision_score(y, y_hat, average='macro'))
print('Recall   :', recall_score(y, y_hat, average='macro'))
print('F1-score :', f1_score(y, y_hat, average='macro'))