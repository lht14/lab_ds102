# LAB 2 – LOGISTIC REGRESSION & SOFTMAX REGRESSION
## 1. Bộ dữ liệu

### 1.1. Nguồn dữ liệu

- **Nguồn:** World Bank Open Data.
- **Giai đoạn:** 2000–2024.
- **Số lượng đặc trưng:** 14.
- **Tập huấn luyện (Train):** 5.015 mẫu.
- **Tập kiểm tra (Test):** 1.253 mẫu.

### 1.2. Các đặc trưng đầu vào

Bộ dữ liệu sử dụng 14 đặc trưng:

1. GDP bình quân đầu người.
2. Tổng GDP.
3. Tỷ lệ lạm phát.
4. Tỷ lệ thất nghiệp.
5. Tỷ lệ thương mại trên GDP.
6. Tỷ lệ FDI trên GDP.
7. Dân số.
8. Tỷ lệ dân số đô thị.
9. Tuổi thọ trung bình.
10. Tỷ lệ tử vong trẻ sơ sinh.
11. Tỷ lệ người dùng Internet.
12. Tỷ lệ tiếp cận điện.
13. Tỷ lệ ghi danh trung học.
14. Lượng phát thải CO2 bình quân đầu người.

### 1.3. Nhãn phân loại

**Logistic Regression – `label_a1`:**

- `0`: Tăng trưởng GDP âm.
- `1`: Tăng trưởng GDP không âm.

**Softmax Regression – `label_a2`:**

- `0`: Tăng trưởng GDP thấp.
- `1`: Tăng trưởng GDP trung bình.
- `2`: Tăng trưởng GDP cao.

## 2. Hướng dẫn chạy chương trình

### 2.1. Cài đặt thư viện

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2.2. Chuẩn bị dữ liệu

Hai tập dữ liệu được lưu trong thư mục `data/`:

- `train_worldbank.csv`
- `test_worldbank.csv`