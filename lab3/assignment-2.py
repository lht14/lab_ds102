from linear_regression import LinearRegression
import pandas as pd

def preprocess_data(df: pd.DataFrame, mean, std):
    continues_variable = [
            'tong_gdp', 'ty_le_lam_phat', 'ty_le_that_nghiep', 'ty_le_thuong_mai_tren_gdp', 'ty_le_fdi_tren_gdp',
            'dan_so', 'ty_le_dan_so_do_thi', 'tuoi_tho_trung_binh', 'ty_le_tu_vong_tre_so_sinh', 'ty_le_nguoi_dung_internet', 'ty_le_tiep_can_dien',
            'ty_le_nhap_hoc_trung_hoc', 'luong_phat_thai_co2_binh_quan_dau_nguoi'
        ]
    for variable in continues_variable:
        df[variable] = (df[variable] - mean[variable]) / std[variable]
    return df

if __name__ == "__main__":
    train = pd.read_csv("lab_ds102/lab3/data/train_worldbank.csv")
    test = pd.read_csv("lab_ds102/lab3/data/test_worldbank.csv")

    features = [
        'tong_gdp', 'ty_le_lam_phat', 'ty_le_that_nghiep', 'ty_le_thuong_mai_tren_gdp', 'ty_le_fdi_tren_gdp',
        'dan_so', 'ty_le_dan_so_do_thi', 'tuoi_tho_trung_binh', 'ty_le_tu_vong_tre_so_sinh', 'ty_le_nguoi_dung_internet', 'ty_le_tiep_can_dien',
        'ty_le_nhap_hoc_trung_hoc', 'luong_phat_thai_co2_binh_quan_dau_nguoi'
    ]

    target = "gdp_binh_quan_dau_nguoi"

    X_train = train[features].to_numpy()
    y_train = (train[target].to_numpy())

    X_test = test[features].to_numpy()
    y = test[target].to_numpy()

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test)
    rmse_original = lr_model.evaluate(y, y_predicted)

    # Chuẩn hóa dữ liệu
    mean = train[features].mean()
    std = train[features].std(ddof=0)

    train = preprocess_data(train, mean, std)
    test = preprocess_data(test, mean, std)

    X_train = train[features].to_numpy()
    y_train = train[target].to_numpy()

    X_test = test[features].to_numpy()
    y = test[target].to_numpy()

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test)
    rmse_normalized = lr_model.evaluate(y, y_predicted)

    print("The root mean squared error is: ", rmse_original)
    print("The root mean squared error (normalized) is: ", rmse_normalized)