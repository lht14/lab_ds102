# Sau khi loại bỏ 2 đặc trưng có tương quan cao, RMSE tăng từ 0.868729 lên 0.938483,
# Nên cho thấy việc loại bỏ đặc trưng chưa cải thiện hiệu quả dự đoán của mô hình Linear Regression trên bộ dữ liệu này

from linear_regression import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def draw_heatmap(df: pd.DataFrame, selected_variables):
    selected_X = df[selected_variables].to_numpy()
    corr = np.corrcoef(selected_X.T)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, xticklabels=selected_variables, yticklabels=selected_variables, annot=True, fmt=".2f", cmap="coolwarm")

    plt.title("Correlation Heatmap")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")
    plt.show() 
    plt.close()
    return corr

if __name__ == "__main__":
    train = pd.read_csv("lab_ds102/lab3/data/train_worldbank.csv")
    test = pd.read_csv("lab_ds102/lab3/data/test_worldbank.csv")

    features = [
        "tong_gdp", "ty_le_lam_phat", "ty_le_that_nghiep", "ty_le_thuong_mai_tren_gdp", "ty_le_fdi_tren_gdp",
        "dan_so", "ty_le_dan_so_do_thi", "tuoi_tho_trung_binh", "ty_le_tu_vong_tre_so_sinh", "ty_le_nguoi_dung_internet", "ty_le_tiep_can_dien",
        "ty_le_nhap_hoc_trung_hoc", "luong_phat_thai_co2_binh_quan_dau_nguoi"
    ]

    target = "gdp_binh_quan_dau_nguoi"

    X_train = train[features].to_numpy()
    y_train = train[target].to_numpy()

    X_test = test[features].to_numpy()
    y = test[target].to_numpy()

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test)
    rmse = lr_model.evaluate(y, y_predicted)

    # Dữ liệu đã qua xử lý
    corr = draw_heatmap(train, features)

    # Loại bỏ các đặc trưng có tương quan cao
    features = ["tong_gdp", "ty_le_lam_phat", "ty_le_that_nghiep", "ty_le_thuong_mai_tren_gdp", "ty_le_fdi_tren_gdp", "dan_so",
    "ty_le_dan_so_do_thi", "ty_le_nguoi_dung_internet", "ty_le_nhap_hoc_trung_hoc", "luong_phat_thai_co2_binh_quan_dau_nguoi", "tuoi_tho_trung_binh"
]

    X_train = train[features].to_numpy()
    X_test = test[features].to_numpy()

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test)
    rmse_after = lr_model.evaluate(y, y_predicted)

    print("RMSE sau khi xử lý: ", rmse_after)
    print("RMSE trước khi xử lý: ", rmse)