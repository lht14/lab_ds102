import pandas as pd
import requests

INDICATORS = {
    "NY.GDP.PCAP.KD": "gdp_binh_quan_dau_nguoi",
    "NY.GDP.MKTP.KD.ZG": "toc_do_tang_truong_gdp",
    "NY.GDP.MKTP.CD": "tong_gdp",
    "FP.CPI.TOTL.ZG": "ty_le_lam_phat",
    "SL.UEM.TOTL.ZS": "ty_le_that_nghiep",
    "NE.TRD.GNFS.ZS": "ty_le_thuong_mai_tren_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "ty_le_fdi_tren_gdp",
    "SP.POP.TOTL": "dan_so",
    "SP.URB.TOTL.IN.ZS": "ty_le_dan_so_do_thi",
    "SP.DYN.LE00.IN": "tuoi_tho_trung_binh",
    "SP.DYN.IMRT.IN": "ty_le_tu_vong_tre_so_sinh",
    "IT.NET.USER.ZS": "ty_le_nguoi_dung_internet",
    "EG.ELC.ACCS.ZS": "ty_le_tiep_can_dien",
    "SE.SEC.ENRR": "ty_le_nhap_hoc_trung_hoc",
    "EN.GHG.CO2.PC.CE.AR5": "luong_phat_thai_co2_binh_quan_dau_nguoi"
}

data = {}

for code, name in INDICATORS.items():
    url = f"https://api.worldbank.org/v2/country/all/indicator/{code}"
    params = {"format": "json", "date": "2000:2024", "per_page": 20000}

    result = requests.get(url, params=params).json()[1]

    for item in result:
        country = item["countryiso3code"]
        year = item["date"]

        if country and item["value"] is not None:
            key = (country, year)

            if key not in data:
                data[key] = {
                    "country_code": country,
                    "country_name": item["country"]["value"],
                    "year": int(year)
                }

            data[key][name] = item["value"]

df = pd.DataFrame(data.values())
df.to_csv("D:/Cpp/lab_ds102/lab1/worldbank_2000_2024.csv")

print("Đã lưu file CSV:", df.shape)