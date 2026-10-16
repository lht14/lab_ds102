import time
from pathlib import Path
import pandas as pd
import requests

BASE_URL = "https://api.worldbank.org/v2"
START_YEAR = 2000
END_YEAR = 2024

INDICATORS = {
    "NY.GDP.PCAP.KD": "gdp_per_capita",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "NY.GDP.MKTP.CD": "gdp_total",
    "FP.CPI.TOTL.ZG": "inflation",
    "SL.UEM.TOTL.ZS": "unemployment_rate",
    "NE.TRD.GNFS.ZS": "trade_gdp_ratio",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_inflow_gdp",
    "SP.POP.TOTL": "population_total",
    "SP.URB.TOTL.IN.ZS": "urban_population_ratio",
    "SP.DYN.LE00.IN": "life_expectancy",
    "SP.DYN.IMRT.IN": "infant_mortality",
    "IT.NET.USER.ZS": "internet_users_ratio",
    "EG.ELC.ACCS.ZS": "electricity_access_ratio",
    "SE.SEC.ENRR": "secondary_school_enrollment",
    "EN.GHG.CO2.PC.CE.AR5": "co2_emissions",
}

def call_api(url, params):
    for i in range(1, 4):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Lần thử {i} thất bại: {e}")
            time.sleep(2)

    return None


def validate_api_data(data):
    if not isinstance(data, list) or len(data) < 2:
        return False
    if not isinstance(data[0], dict) or "pages" not in data[0]:
        return False
    return True

def get_countries():
    countries = {}
    page = 1
    total_pages = 1

    while page <= total_pages:
        data = call_api(BASE_URL + "/country", {"format": "json", "per_page": 300, "page": page})

        if not validate_api_data(data):
            print("Không lấy được danh sách quốc gia/vùng lãnh thổ")
            return {}
        total_pages = int(data[0]["pages"])

        for item in data[1]:
            # Không lấy mã NA
            if item["region"]["id"] != "NA":
                countries[item["id"]] = item["name"]
        page = page + 1

    print("Tổng số quốc gia/vùng lãnh thổ: ", len(countries))
    return countries

def get_indicator(indicator_code, countries):
    url = BASE_URL + "/country/all/indicator/" + indicator_code

    results = {}
    page = 1
    total_pages = 1

    while page <= total_pages:
        params = {
            "format": "json",
            "date": str(START_YEAR) + ":" + str(END_YEAR),
            "per_page": 1000,
            "page": page,
        }

        data = call_api(url, params)
        if not validate_api_data(data):
            page = page + 1
            if page == 2 and total_pages == 1:
                break
            continue

        total_pages = int(data[0]["pages"])
        for item in data[1]:
            country_code = item["countryiso3code"]
            year = item["date"]
            value = item["value"]

            if country_code in countries and value is not None:
                results[country_code + "_" + year] = value

        page = page + 1
        time.sleep(0.3)

    return results


def main():
    countries = get_countries()

    if len(countries) == 0:
        return

    dataset = {}
    index = 1

    for indicator_code, column_name in INDICATORS.items():
        dataset[column_name] = get_indicator(indicator_code, countries)
        index = index + 1

    rows = []

    for country_code, country_name in countries.items():
        for year in range(START_YEAR, END_YEAR + 1):
            row = {
                "country_code": country_code,
                "country_name": country_name,
                "year": year,
            }

            for column_name in INDICATORS.values():
                key = country_code + "_" + str(year)
                row[column_name] = dataset[column_name].get(key)

            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["country_code", "year"])

    df.to_csv("D:/Cpp/lab_ds102/lab1/worldbank_2000_2024.csv", index=False, encoding="utf-8-sig")
if __name__ == "__main__":
    main()