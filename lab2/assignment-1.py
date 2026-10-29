import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from LogisticRegression import LogisticRegression

train_data = pd.read_csv("lab_ds102/lab2/data/train_worldbank.csv")
test_data = pd.read_csv("lab_ds102/lab2/data/test_worldbank.csv")

# 14 thuộc tính đầu vào; sử dụng label_a1
features = [
    'gdp_binh_quan_dau_nguoi', 'tong_gdp', 'ty_le_lam_phat', 'ty_le_that_nghiep', 'ty_le_thuong_mai_tren_gdp', 'ty_le_fdi_tren_gdp',
    'dan_so', 'ty_le_dan_so_do_thi', 'tuoi_tho_trung_binh', 'ty_le_tu_vong_tre_so_sinh', 'ty_le_nguoi_dung_internet', 'ty_le_tiep_can_dien',
    'ty_le_nhap_hoc_trung_hoc', 'luong_phat_thai_co2_binh_quan_dau_nguoi'
]
train_data = train_data[train_data['label_a1'].isin([0, 1])]
train_X = train_data[features].apply(pd.to_numeric, errors='coerce')
train_labels = train_data['label_a1'].to_numpy(dtype=np.float64)

test_data = test_data[test_data['label_a1'].isin([0, 1])]
test_X = test_data[features].apply(pd.to_numeric, errors='coerce')
test_labels = test_data['label_a1'].to_numpy(dtype=np.float64)

# Chuẩn hóa
means = train_X.mean()
stds = train_X.std().replace(0, 1).fillna(1)
train_X = ((train_X - means) / stds).to_numpy(dtype=np.float64)
test_X = ((test_X - means) / stds).to_numpy(dtype=np.float64)

model = LogisticRegression(
    epoch=100,
    lr=0.1
)
model.fit(train_X, train_labels)
predicted_labels = model.predict(test_X)
metrics = model.evaluate(test_labels, predicted_labels)
print('=' * 10)

for metric in metrics:
    print(f'{metric}: {metrics[metric]}')

# Chuyển xác suất thành nhãn
predicted_classes = np.where(predicted_labels <= 0.5, 0, 1)
cfm = confusion_matrix(test_labels, predicted_classes)
print(cfm)
