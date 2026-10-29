import numpy as np
import pandas as pd
from SoftmaxRegression import SoftmaxRegression
from matplotlib import pyplot as plt

def convert_to_one_hot(labels: np.ndarray):
    one_hot_labels = np.zeros((labels.shape[0], 3))
    one_hot_labels[np.arange(labels.shape[0]), labels] = 1
    return one_hot_labels

train_data = pd.read_csv("lab_ds102/lab2/data/train_worldbank.csv")
test_data = pd.read_csv("lab_ds102/lab2/data/test_worldbank.csv")

features = [
    'gdp_binh_quan_dau_nguoi', 'tong_gdp', 'ty_le_lam_phat', 'ty_le_that_nghiep', 'ty_le_thuong_mai_tren_gdp', 'ty_le_fdi_tren_gdp',
    'dan_so', 'ty_le_dan_so_do_thi', 'tuoi_tho_trung_binh', 'ty_le_tu_vong_tre_so_sinh', 'ty_le_nguoi_dung_internet', 'ty_le_tiep_can_dien',
    'ty_le_nhap_hoc_trung_hoc', 'luong_phat_thai_co2_binh_quan_dau_nguoi'
]

# Lấy nhãn labels_a2 (0: thấp, 1: trung bình, 2: cao)
train_data = train_data[train_data['label_a2'].isin([0, 1, 2])]
test_data = test_data[test_data['label_a2'].isin([0, 1, 2])]
train_labels = train_data['label_a2'].to_numpy(dtype=int)
encoded_train_labels = convert_to_one_hot(train_labels)

test_labels = test_data['label_a2'].to_numpy(dtype=int)
encoded_test_labels = convert_to_one_hot(test_labels)

train_X = train_data[features].apply(pd.to_numeric, errors='coerce')
test_X = test_data[features].apply(pd.to_numeric, errors='coerce')

# Chuẩn hóa
means = train_X.mean()
stds = train_X.std().replace(0, 1).fillna(1)
train_X = ((train_X - means) / stds).to_numpy(dtype=np.float64)
test_X = ((test_X - means) / stds).to_numpy(dtype=np.float64)

model = SoftmaxRegression(
    epoch=500,
    lr=0.01
)
model.fit(train_X, encoded_train_labels)
predicted_labels = model.predict(test_X)
results = model.evaluate(test_labels, predicted_labels)
print(results)

plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss")
plt.grid()
plt.show()