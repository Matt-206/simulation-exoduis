"""
Prefecture-level demographic calibration data for Japan.

All data sourced from:
  - 2020 Population Census (e-Stat, Statistics Bureau of Japan)
  - 2020 School Basic Survey (MEXT) for education attainment
  - 2020 Vital Statistics (MHLW) for TFR and marriage
  - 2020 Basic Survey on Wage Structure (MHLW) for income
  - 2022 Housing and Land Survey for housing vacancy (Akiya)

Each prefecture entry contains:
  age_distribution: dict of age-group shares (sums to 1.0)
  median_age: float
  pct_female: float (overall share of female population)
  university_rate: float (% of 25+ with university degree)
  hs_rate: float (% of 25+ with only high-school diploma)
  mean_income_jpy: float (annual mean income)
  marriage_rate_adj: float (multiplier vs national average; >1 = more marriage)
  mean_age_first_marriage_m: float
  mean_age_first_marriage_f: float
  tfr: float (total fertility rate, 2020)
  pct_single_30_34: float (% never-married among 30-34 year-olds)
  vacancy_rate: float (housing vacancy rate)
"""

PREFECTURE_DEMOGRAPHICS = {
    "Hokkaido": {
        "age_distribution": {"0-14": 0.103, "15-19": 0.039, "20-24": 0.042, "25-29": 0.044, "30-34": 0.047, "35-39": 0.053, "40-44": 0.058, "45-49": 0.070, "50-54": 0.073, "55-59": 0.066, "60-64": 0.062, "65-69": 0.072, "70-74": 0.085, "75-79": 0.069, "80-84": 0.053, "85+": 0.064},
        "median_age": 50.4, "pct_female": 0.527, "university_rate": 0.30, "hs_rate": 0.38,
        "mean_income_jpy": 3_850_000, "marriage_rate_adj": 0.88, "mean_age_first_marriage_m": 30.7, "mean_age_first_marriage_f": 29.2,
        "tfr": 1.21, "pct_single_30_34": 0.48, "vacancy_rate": 0.154,
    },
    "Aomori": {
        "age_distribution": {"0-14": 0.098, "15-19": 0.040, "20-24": 0.038, "25-29": 0.039, "30-34": 0.042, "35-39": 0.049, "40-44": 0.056, "45-49": 0.069, "50-54": 0.073, "55-59": 0.068, "60-64": 0.066, "65-69": 0.078, "70-74": 0.092, "75-79": 0.073, "80-84": 0.056, "85+": 0.063},
        "median_age": 52.0, "pct_female": 0.525, "university_rate": 0.22, "hs_rate": 0.42,
        "mean_income_jpy": 3_560_000, "marriage_rate_adj": 0.90, "mean_age_first_marriage_m": 30.4, "mean_age_first_marriage_f": 28.8,
        "tfr": 1.33, "pct_single_30_34": 0.44, "vacancy_rate": 0.167,
    },
    "Iwate": {
        "age_distribution": {"0-14": 0.100, "15-19": 0.041, "20-24": 0.037, "25-29": 0.039, "30-34": 0.043, "35-39": 0.050, "40-44": 0.056, "45-49": 0.069, "50-54": 0.072, "55-59": 0.067, "60-64": 0.065, "65-69": 0.077, "70-74": 0.092, "75-79": 0.074, "80-84": 0.056, "85+": 0.062},
        "median_age": 52.2, "pct_female": 0.521, "university_rate": 0.22, "hs_rate": 0.42,
        "mean_income_jpy": 3_640_000, "marriage_rate_adj": 0.92, "mean_age_first_marriage_m": 30.3, "mean_age_first_marriage_f": 28.7,
        "tfr": 1.33, "pct_single_30_34": 0.42, "vacancy_rate": 0.162,
    },
    "Miyagi": {
        "age_distribution": {"0-14": 0.112, "15-19": 0.043, "20-24": 0.048, "25-29": 0.050, "30-34": 0.053, "35-39": 0.057, "40-44": 0.061, "45-49": 0.071, "50-54": 0.072, "55-59": 0.065, "60-64": 0.060, "65-69": 0.069, "70-74": 0.080, "75-79": 0.062, "80-84": 0.046, "85+": 0.051},
        "median_age": 48.0, "pct_female": 0.515, "university_rate": 0.34, "hs_rate": 0.36,
        "mean_income_jpy": 4_100_000, "marriage_rate_adj": 0.95, "mean_age_first_marriage_m": 30.7, "mean_age_first_marriage_f": 29.1,
        "tfr": 1.21, "pct_single_30_34": 0.45, "vacancy_rate": 0.138,
    },
    "Akita": {
        "age_distribution": {"0-14": 0.085, "15-19": 0.037, "20-24": 0.034, "25-29": 0.035, "30-34": 0.038, "35-39": 0.045, "40-44": 0.052, "45-49": 0.065, "50-54": 0.071, "55-59": 0.068, "60-64": 0.068, "65-69": 0.083, "70-74": 0.100, "75-79": 0.081, "80-84": 0.064, "85+": 0.074},
        "median_age": 55.2, "pct_female": 0.528, "university_rate": 0.20, "hs_rate": 0.44,
        "mean_income_jpy": 3_480_000, "marriage_rate_adj": 0.88, "mean_age_first_marriage_m": 30.5, "mean_age_first_marriage_f": 28.9,
        "tfr": 1.24, "pct_single_30_34": 0.40, "vacancy_rate": 0.173,
    },
    "Yamagata": {
        "age_distribution": {"0-14": 0.103, "15-19": 0.042, "20-24": 0.037, "25-29": 0.039, "30-34": 0.043, "35-39": 0.050, "40-44": 0.056, "45-49": 0.068, "50-54": 0.071, "55-59": 0.066, "60-64": 0.064, "65-69": 0.077, "70-74": 0.091, "75-79": 0.075, "80-84": 0.057, "85+": 0.061},
        "median_age": 52.1, "pct_female": 0.520, "university_rate": 0.21, "hs_rate": 0.43,
        "mean_income_jpy": 3_680_000, "marriage_rate_adj": 0.98, "mean_age_first_marriage_m": 30.0, "mean_age_first_marriage_f": 28.5,
        "tfr": 1.40, "pct_single_30_34": 0.36, "vacancy_rate": 0.145,
    },
    "Fukushima": {
        "age_distribution": {"0-14": 0.105, "15-19": 0.042, "20-24": 0.040, "25-29": 0.042, "30-34": 0.046, "35-39": 0.052, "40-44": 0.058, "45-49": 0.069, "50-54": 0.072, "55-59": 0.066, "60-64": 0.063, "65-69": 0.075, "70-74": 0.088, "75-79": 0.070, "80-84": 0.052, "85+": 0.060},
        "median_age": 50.8, "pct_female": 0.517, "university_rate": 0.23, "hs_rate": 0.42,
        "mean_income_jpy": 3_820_000, "marriage_rate_adj": 0.94, "mean_age_first_marriage_m": 30.2, "mean_age_first_marriage_f": 28.6,
        "tfr": 1.38, "pct_single_30_34": 0.39, "vacancy_rate": 0.148,
    },
    "Ibaraki": {
        "age_distribution": {"0-14": 0.113, "15-19": 0.043, "20-24": 0.044, "25-29": 0.047, "30-34": 0.051, "35-39": 0.056, "40-44": 0.062, "45-49": 0.073, "50-54": 0.074, "55-59": 0.066, "60-64": 0.061, "65-69": 0.069, "70-74": 0.081, "75-79": 0.063, "80-84": 0.046, "85+": 0.051},
        "median_age": 48.8, "pct_female": 0.510, "university_rate": 0.33, "hs_rate": 0.36,
        "mean_income_jpy": 4_280_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 30.6, "mean_age_first_marriage_f": 28.9,
        "tfr": 1.36, "pct_single_30_34": 0.42, "vacancy_rate": 0.140,
    },
    "Tochigi": {
        "age_distribution": {"0-14": 0.113, "15-19": 0.043, "20-24": 0.043, "25-29": 0.047, "30-34": 0.051, "35-39": 0.056, "40-44": 0.062, "45-49": 0.072, "50-54": 0.073, "55-59": 0.066, "60-64": 0.062, "65-69": 0.070, "70-74": 0.082, "75-79": 0.064, "80-84": 0.047, "85+": 0.049},
        "median_age": 48.7, "pct_female": 0.509, "university_rate": 0.30, "hs_rate": 0.38,
        "mean_income_jpy": 4_180_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 30.3, "mean_age_first_marriage_f": 28.7,
        "tfr": 1.37, "pct_single_30_34": 0.40, "vacancy_rate": 0.139,
    },
    "Gunma": {
        "age_distribution": {"0-14": 0.112, "15-19": 0.043, "20-24": 0.043, "25-29": 0.047, "30-34": 0.051, "35-39": 0.056, "40-44": 0.062, "45-49": 0.072, "50-54": 0.073, "55-59": 0.066, "60-64": 0.062, "65-69": 0.071, "70-74": 0.083, "75-79": 0.065, "80-84": 0.048, "85+": 0.046},
        "median_age": 49.0, "pct_female": 0.511, "university_rate": 0.29, "hs_rate": 0.38,
        "mean_income_jpy": 4_100_000, "marriage_rate_adj": 0.99, "mean_age_first_marriage_m": 30.4, "mean_age_first_marriage_f": 28.8,
        "tfr": 1.38, "pct_single_30_34": 0.40, "vacancy_rate": 0.142,
    },
    "Saitama": {
        "age_distribution": {"0-14": 0.119, "15-19": 0.044, "20-24": 0.048, "25-29": 0.053, "30-34": 0.056, "35-39": 0.060, "40-44": 0.065, "45-49": 0.075, "50-54": 0.074, "55-59": 0.064, "60-64": 0.057, "65-69": 0.065, "70-74": 0.076, "75-79": 0.058, "80-84": 0.042, "85+": 0.044},
        "median_age": 47.0, "pct_female": 0.510, "university_rate": 0.40, "hs_rate": 0.32,
        "mean_income_jpy": 4_540_000, "marriage_rate_adj": 1.02, "mean_age_first_marriage_m": 31.3, "mean_age_first_marriage_f": 29.6,
        "tfr": 1.22, "pct_single_30_34": 0.46, "vacancy_rate": 0.103,
    },
    "Chiba": {
        "age_distribution": {"0-14": 0.116, "15-19": 0.043, "20-24": 0.047, "25-29": 0.051, "30-34": 0.055, "35-39": 0.059, "40-44": 0.064, "45-49": 0.074, "50-54": 0.074, "55-59": 0.064, "60-64": 0.058, "65-69": 0.066, "70-74": 0.078, "75-79": 0.060, "80-84": 0.044, "85+": 0.047},
        "median_age": 47.8, "pct_female": 0.511, "university_rate": 0.39, "hs_rate": 0.32,
        "mean_income_jpy": 4_480_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 31.2, "mean_age_first_marriage_f": 29.5,
        "tfr": 1.21, "pct_single_30_34": 0.46, "vacancy_rate": 0.123,
    },
    "Tokyo": {
        "age_distribution": {"0-14": 0.107, "15-19": 0.037, "20-24": 0.060, "25-29": 0.072, "30-34": 0.071, "35-39": 0.069, "40-44": 0.067, "45-49": 0.075, "50-54": 0.073, "55-59": 0.060, "60-64": 0.049, "65-69": 0.055, "70-74": 0.064, "75-79": 0.052, "80-84": 0.040, "85+": 0.049},
        "median_age": 45.2, "pct_female": 0.514, "university_rate": 0.65, "hs_rate": 0.18,
        "mean_income_jpy": 6_220_000, "marriage_rate_adj": 0.82, "mean_age_first_marriage_m": 32.3, "mean_age_first_marriage_f": 30.5,
        "tfr": 1.08, "pct_single_30_34": 0.56, "vacancy_rate": 0.108,
    },
    "Kanagawa": {
        "age_distribution": {"0-14": 0.117, "15-19": 0.042, "20-24": 0.050, "25-29": 0.056, "30-34": 0.058, "35-39": 0.062, "40-44": 0.066, "45-49": 0.076, "50-54": 0.074, "55-59": 0.063, "60-64": 0.055, "65-69": 0.062, "70-74": 0.073, "75-79": 0.057, "80-84": 0.042, "85+": 0.047},
        "median_age": 46.5, "pct_female": 0.509, "university_rate": 0.46, "hs_rate": 0.28,
        "mean_income_jpy": 5_080_000, "marriage_rate_adj": 0.95, "mean_age_first_marriage_m": 31.7, "mean_age_first_marriage_f": 29.9,
        "tfr": 1.19, "pct_single_30_34": 0.50, "vacancy_rate": 0.105,
    },
    "Niigata": {
        "age_distribution": {"0-14": 0.103, "15-19": 0.042, "20-24": 0.040, "25-29": 0.042, "30-34": 0.046, "35-39": 0.052, "40-44": 0.058, "45-49": 0.069, "50-54": 0.072, "55-59": 0.066, "60-64": 0.063, "65-69": 0.076, "70-74": 0.090, "75-79": 0.072, "80-84": 0.054, "85+": 0.055},
        "median_age": 51.4, "pct_female": 0.519, "university_rate": 0.26, "hs_rate": 0.40,
        "mean_income_jpy": 3_920_000, "marriage_rate_adj": 0.96, "mean_age_first_marriage_m": 30.1, "mean_age_first_marriage_f": 28.5,
        "tfr": 1.32, "pct_single_30_34": 0.39, "vacancy_rate": 0.148,
    },
    "Toyama": {
        "age_distribution": {"0-14": 0.107, "15-19": 0.042, "20-24": 0.040, "25-29": 0.044, "30-34": 0.048, "35-39": 0.053, "40-44": 0.058, "45-49": 0.069, "50-54": 0.071, "55-59": 0.065, "60-64": 0.062, "65-69": 0.074, "70-74": 0.088, "75-79": 0.070, "80-84": 0.053, "85+": 0.056},
        "median_age": 50.6, "pct_female": 0.518, "university_rate": 0.30, "hs_rate": 0.38,
        "mean_income_jpy": 4_160_000, "marriage_rate_adj": 1.04, "mean_age_first_marriage_m": 30.1, "mean_age_first_marriage_f": 28.5,
        "tfr": 1.42, "pct_single_30_34": 0.35, "vacancy_rate": 0.132,
    },
    "Ishikawa": {
        "age_distribution": {"0-14": 0.113, "15-19": 0.043, "20-24": 0.045, "25-29": 0.048, "30-34": 0.051, "35-39": 0.055, "40-44": 0.059, "45-49": 0.069, "50-54": 0.070, "55-59": 0.064, "60-64": 0.061, "65-69": 0.072, "70-74": 0.084, "75-79": 0.066, "80-84": 0.050, "85+": 0.050},
        "median_age": 49.0, "pct_female": 0.517, "university_rate": 0.36, "hs_rate": 0.34,
        "mean_income_jpy": 4_100_000, "marriage_rate_adj": 1.02, "mean_age_first_marriage_m": 30.3, "mean_age_first_marriage_f": 28.8,
        "tfr": 1.42, "pct_single_30_34": 0.38, "vacancy_rate": 0.137,
    },
    "Fukui": {
        "age_distribution": {"0-14": 0.117, "15-19": 0.044, "20-24": 0.040, "25-29": 0.044, "30-34": 0.049, "35-39": 0.054, "40-44": 0.058, "45-49": 0.068, "50-54": 0.070, "55-59": 0.064, "60-64": 0.062, "65-69": 0.073, "70-74": 0.086, "75-79": 0.068, "80-84": 0.051, "85+": 0.052},
        "median_age": 49.5, "pct_female": 0.518, "university_rate": 0.28, "hs_rate": 0.39,
        "mean_income_jpy": 4_080_000, "marriage_rate_adj": 1.06, "mean_age_first_marriage_m": 29.9, "mean_age_first_marriage_f": 28.3,
        "tfr": 1.56, "pct_single_30_34": 0.33, "vacancy_rate": 0.143,
    },
    "Yamanashi": {
        "age_distribution": {"0-14": 0.108, "15-19": 0.042, "20-24": 0.042, "25-29": 0.045, "30-34": 0.049, "35-39": 0.054, "40-44": 0.060, "45-49": 0.071, "50-54": 0.073, "55-59": 0.066, "60-64": 0.063, "65-69": 0.072, "70-74": 0.085, "75-79": 0.067, "80-84": 0.050, "85+": 0.053},
        "median_age": 49.8, "pct_female": 0.512, "university_rate": 0.28, "hs_rate": 0.38,
        "mean_income_jpy": 3_980_000, "marriage_rate_adj": 0.96, "mean_age_first_marriage_m": 30.6, "mean_age_first_marriage_f": 28.9,
        "tfr": 1.36, "pct_single_30_34": 0.41, "vacancy_rate": 0.215,
    },
    "Nagano": {
        "age_distribution": {"0-14": 0.110, "15-19": 0.043, "20-24": 0.039, "25-29": 0.042, "30-34": 0.047, "35-39": 0.053, "40-44": 0.058, "45-49": 0.069, "50-54": 0.071, "55-59": 0.065, "60-64": 0.063, "65-69": 0.075, "70-74": 0.088, "75-79": 0.070, "80-84": 0.053, "85+": 0.054},
        "median_age": 50.5, "pct_female": 0.515, "university_rate": 0.27, "hs_rate": 0.40,
        "mean_income_jpy": 3_960_000, "marriage_rate_adj": 0.98, "mean_age_first_marriage_m": 30.2, "mean_age_first_marriage_f": 28.7,
        "tfr": 1.44, "pct_single_30_34": 0.38, "vacancy_rate": 0.194,
    },
    "Gifu": {
        "age_distribution": {"0-14": 0.115, "15-19": 0.044, "20-24": 0.042, "25-29": 0.046, "30-34": 0.050, "35-39": 0.055, "40-44": 0.061, "45-49": 0.072, "50-54": 0.073, "55-59": 0.065, "60-64": 0.061, "65-69": 0.070, "70-74": 0.082, "75-79": 0.064, "80-84": 0.048, "85+": 0.052},
        "median_age": 49.2, "pct_female": 0.514, "university_rate": 0.28, "hs_rate": 0.39,
        "mean_income_jpy": 4_080_000, "marriage_rate_adj": 1.02, "mean_age_first_marriage_m": 30.0, "mean_age_first_marriage_f": 28.4,
        "tfr": 1.40, "pct_single_30_34": 0.38, "vacancy_rate": 0.150,
    },
    "Shizuoka": {
        "age_distribution": {"0-14": 0.114, "15-19": 0.044, "20-24": 0.043, "25-29": 0.047, "30-34": 0.051, "35-39": 0.056, "40-44": 0.061, "45-49": 0.072, "50-54": 0.073, "55-59": 0.066, "60-64": 0.062, "65-69": 0.070, "70-74": 0.082, "75-79": 0.063, "80-84": 0.047, "85+": 0.049},
        "median_age": 49.1, "pct_female": 0.511, "university_rate": 0.30, "hs_rate": 0.38,
        "mean_income_jpy": 4_200_000, "marriage_rate_adj": 1.02, "mean_age_first_marriage_m": 30.2, "mean_age_first_marriage_f": 28.6,
        "tfr": 1.36, "pct_single_30_34": 0.40, "vacancy_rate": 0.147,
    },
    "Aichi": {
        "age_distribution": {"0-14": 0.130, "15-19": 0.046, "20-24": 0.051, "25-29": 0.055, "30-34": 0.058, "35-39": 0.061, "40-44": 0.064, "45-49": 0.073, "50-54": 0.072, "55-59": 0.063, "60-64": 0.056, "65-69": 0.064, "70-74": 0.072, "75-79": 0.054, "80-84": 0.039, "85+": 0.042},
        "median_age": 45.8, "pct_female": 0.505, "university_rate": 0.40, "hs_rate": 0.32,
        "mean_income_jpy": 4_820_000, "marriage_rate_adj": 1.05, "mean_age_first_marriage_m": 30.5, "mean_age_first_marriage_f": 28.7,
        "tfr": 1.38, "pct_single_30_34": 0.40, "vacancy_rate": 0.118,
    },
    "Mie": {
        "age_distribution": {"0-14": 0.114, "15-19": 0.043, "20-24": 0.042, "25-29": 0.046, "30-34": 0.050, "35-39": 0.055, "40-44": 0.061, "45-49": 0.072, "50-54": 0.073, "55-59": 0.066, "60-64": 0.062, "65-69": 0.071, "70-74": 0.083, "75-79": 0.064, "80-84": 0.048, "85+": 0.050},
        "median_age": 49.3, "pct_female": 0.513, "university_rate": 0.28, "hs_rate": 0.38,
        "mean_income_jpy": 4_100_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 30.1, "mean_age_first_marriage_f": 28.5,
        "tfr": 1.37, "pct_single_30_34": 0.38, "vacancy_rate": 0.147,
    },
    "Shiga": {
        "age_distribution": {"0-14": 0.135, "15-19": 0.048, "20-24": 0.048, "25-29": 0.052, "30-34": 0.057, "35-39": 0.061, "40-44": 0.064, "45-49": 0.073, "50-54": 0.072, "55-59": 0.063, "60-64": 0.056, "65-69": 0.064, "70-74": 0.072, "75-79": 0.054, "80-84": 0.039, "85+": 0.042},
        "median_age": 45.5, "pct_female": 0.508, "university_rate": 0.35, "hs_rate": 0.34,
        "mean_income_jpy": 4_320_000, "marriage_rate_adj": 1.06, "mean_age_first_marriage_m": 30.0, "mean_age_first_marriage_f": 28.4,
        "tfr": 1.50, "pct_single_30_34": 0.36, "vacancy_rate": 0.131,
    },
    "Kyoto": {
        "age_distribution": {"0-14": 0.109, "15-19": 0.042, "20-24": 0.055, "25-29": 0.054, "30-34": 0.054, "35-39": 0.058, "40-44": 0.062, "45-49": 0.072, "50-54": 0.072, "55-59": 0.064, "60-64": 0.058, "65-69": 0.067, "70-74": 0.078, "75-79": 0.062, "80-84": 0.047, "85+": 0.046},
        "median_age": 47.8, "pct_female": 0.520, "university_rate": 0.44, "hs_rate": 0.28,
        "mean_income_jpy": 4_540_000, "marriage_rate_adj": 0.90, "mean_age_first_marriage_m": 31.2, "mean_age_first_marriage_f": 29.5,
        "tfr": 1.18, "pct_single_30_34": 0.48, "vacancy_rate": 0.131,
    },
    "Osaka": {
        "age_distribution": {"0-14": 0.114, "15-19": 0.041, "20-24": 0.052, "25-29": 0.057, "30-34": 0.058, "35-39": 0.062, "40-44": 0.065, "45-49": 0.074, "50-54": 0.073, "55-59": 0.063, "60-64": 0.056, "65-69": 0.063, "70-74": 0.073, "75-79": 0.058, "80-84": 0.043, "85+": 0.048},
        "median_age": 46.8, "pct_female": 0.518, "university_rate": 0.40, "hs_rate": 0.30,
        "mean_income_jpy": 4_780_000, "marriage_rate_adj": 0.92, "mean_age_first_marriage_m": 31.0, "mean_age_first_marriage_f": 29.3,
        "tfr": 1.22, "pct_single_30_34": 0.48, "vacancy_rate": 0.152,
    },
    "Hyogo": {
        "age_distribution": {"0-14": 0.117, "15-19": 0.043, "20-24": 0.047, "25-29": 0.051, "30-34": 0.054, "35-39": 0.058, "40-44": 0.063, "45-49": 0.073, "50-54": 0.073, "55-59": 0.064, "60-64": 0.058, "65-69": 0.067, "70-74": 0.078, "75-79": 0.062, "80-84": 0.046, "85+": 0.046},
        "median_age": 48.0, "pct_female": 0.519, "university_rate": 0.37, "hs_rate": 0.32,
        "mean_income_jpy": 4_400_000, "marriage_rate_adj": 0.96, "mean_age_first_marriage_m": 30.9, "mean_age_first_marriage_f": 29.2,
        "tfr": 1.29, "pct_single_30_34": 0.44, "vacancy_rate": 0.136,
    },
    "Nara": {
        "age_distribution": {"0-14": 0.109, "15-19": 0.042, "20-24": 0.043, "25-29": 0.046, "30-34": 0.050, "35-39": 0.055, "40-44": 0.061, "45-49": 0.073, "50-54": 0.075, "55-59": 0.066, "60-64": 0.060, "65-69": 0.069, "70-74": 0.082, "75-79": 0.065, "80-84": 0.049, "85+": 0.055},
        "median_age": 49.7, "pct_female": 0.524, "university_rate": 0.35, "hs_rate": 0.33,
        "mean_income_jpy": 4_100_000, "marriage_rate_adj": 0.94, "mean_age_first_marriage_m": 31.0, "mean_age_first_marriage_f": 29.3,
        "tfr": 1.22, "pct_single_30_34": 0.44, "vacancy_rate": 0.146,
    },
    "Wakayama": {
        "age_distribution": {"0-14": 0.105, "15-19": 0.041, "20-24": 0.039, "25-29": 0.042, "30-34": 0.047, "35-39": 0.052, "40-44": 0.058, "45-49": 0.070, "50-54": 0.073, "55-59": 0.067, "60-64": 0.065, "65-69": 0.075, "70-74": 0.088, "75-79": 0.070, "80-84": 0.054, "85+": 0.054},
        "median_age": 51.2, "pct_female": 0.524, "university_rate": 0.25, "hs_rate": 0.40,
        "mean_income_jpy": 3_860_000, "marriage_rate_adj": 0.94, "mean_age_first_marriage_m": 30.2, "mean_age_first_marriage_f": 28.6,
        "tfr": 1.40, "pct_single_30_34": 0.40, "vacancy_rate": 0.202,
    },
    "Tottori": {
        "age_distribution": {"0-14": 0.113, "15-19": 0.044, "20-24": 0.041, "25-29": 0.044, "30-34": 0.048, "35-39": 0.053, "40-44": 0.058, "45-49": 0.068, "50-54": 0.070, "55-59": 0.064, "60-64": 0.063, "65-69": 0.076, "70-74": 0.088, "75-79": 0.069, "80-84": 0.052, "85+": 0.049},
        "median_age": 50.1, "pct_female": 0.519, "university_rate": 0.26, "hs_rate": 0.40,
        "mean_income_jpy": 3_700_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 29.8, "mean_age_first_marriage_f": 28.2,
        "tfr": 1.53, "pct_single_30_34": 0.36, "vacancy_rate": 0.158,
    },
    "Shimane": {
        "age_distribution": {"0-14": 0.110, "15-19": 0.044, "20-24": 0.040, "25-29": 0.042, "30-34": 0.046, "35-39": 0.052, "40-44": 0.056, "45-49": 0.066, "50-54": 0.069, "55-59": 0.064, "60-64": 0.064, "65-69": 0.078, "70-74": 0.092, "75-79": 0.074, "80-84": 0.057, "85+": 0.046},
        "median_age": 51.8, "pct_female": 0.521, "university_rate": 0.23, "hs_rate": 0.42,
        "mean_income_jpy": 3_700_000, "marriage_rate_adj": 1.02, "mean_age_first_marriage_m": 29.7, "mean_age_first_marriage_f": 28.1,
        "tfr": 1.62, "pct_single_30_34": 0.34, "vacancy_rate": 0.155,
    },
    "Okayama": {
        "age_distribution": {"0-14": 0.117, "15-19": 0.044, "20-24": 0.046, "25-29": 0.049, "30-34": 0.052, "35-39": 0.056, "40-44": 0.060, "45-49": 0.070, "50-54": 0.071, "55-59": 0.065, "60-64": 0.062, "65-69": 0.071, "70-74": 0.083, "75-79": 0.065, "80-84": 0.049, "85+": 0.040},
        "median_age": 48.8, "pct_female": 0.517, "university_rate": 0.32, "hs_rate": 0.36,
        "mean_income_jpy": 4_020_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 30.1, "mean_age_first_marriage_f": 28.6,
        "tfr": 1.47, "pct_single_30_34": 0.39, "vacancy_rate": 0.152,
    },
    "Hiroshima": {
        "age_distribution": {"0-14": 0.118, "15-19": 0.044, "20-24": 0.047, "25-29": 0.050, "30-34": 0.053, "35-39": 0.057, "40-44": 0.061, "45-49": 0.071, "50-54": 0.072, "55-59": 0.065, "60-64": 0.060, "65-69": 0.069, "70-74": 0.081, "75-79": 0.063, "80-84": 0.047, "85+": 0.042},
        "median_age": 48.2, "pct_female": 0.515, "university_rate": 0.34, "hs_rate": 0.35,
        "mean_income_jpy": 4_280_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 30.3, "mean_age_first_marriage_f": 28.8,
        "tfr": 1.42, "pct_single_30_34": 0.41, "vacancy_rate": 0.153,
    },
    "Yamaguchi": {
        "age_distribution": {"0-14": 0.105, "15-19": 0.041, "20-24": 0.040, "25-29": 0.043, "30-34": 0.047, "35-39": 0.053, "40-44": 0.058, "45-49": 0.069, "50-54": 0.072, "55-59": 0.066, "60-64": 0.064, "65-69": 0.076, "70-74": 0.090, "75-79": 0.071, "80-84": 0.054, "85+": 0.051},
        "median_age": 51.5, "pct_female": 0.523, "university_rate": 0.27, "hs_rate": 0.39,
        "mean_income_jpy": 3_940_000, "marriage_rate_adj": 0.96, "mean_age_first_marriage_m": 30.0, "mean_age_first_marriage_f": 28.5,
        "tfr": 1.44, "pct_single_30_34": 0.39, "vacancy_rate": 0.168,
    },
    "Tokushima": {
        "age_distribution": {"0-14": 0.103, "15-19": 0.041, "20-24": 0.041, "25-29": 0.043, "30-34": 0.047, "35-39": 0.052, "40-44": 0.058, "45-49": 0.069, "50-54": 0.072, "55-59": 0.066, "60-64": 0.064, "65-69": 0.076, "70-74": 0.089, "75-79": 0.071, "80-84": 0.054, "85+": 0.054},
        "median_age": 51.5, "pct_female": 0.524, "university_rate": 0.28, "hs_rate": 0.38,
        "mean_income_jpy": 3_800_000, "marriage_rate_adj": 0.94, "mean_age_first_marriage_m": 30.0, "mean_age_first_marriage_f": 28.5,
        "tfr": 1.38, "pct_single_30_34": 0.40, "vacancy_rate": 0.192,
    },
    "Kagawa": {
        "age_distribution": {"0-14": 0.112, "15-19": 0.043, "20-24": 0.043, "25-29": 0.046, "30-34": 0.050, "35-39": 0.055, "40-44": 0.060, "45-49": 0.070, "50-54": 0.072, "55-59": 0.066, "60-64": 0.063, "65-69": 0.073, "70-74": 0.085, "75-79": 0.067, "80-84": 0.050, "85+": 0.045},
        "median_age": 50.0, "pct_female": 0.517, "university_rate": 0.29, "hs_rate": 0.38,
        "mean_income_jpy": 3_900_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 30.0, "mean_age_first_marriage_f": 28.5,
        "tfr": 1.50, "pct_single_30_34": 0.38, "vacancy_rate": 0.182,
    },
    "Ehime": {
        "age_distribution": {"0-14": 0.107, "15-19": 0.042, "20-24": 0.042, "25-29": 0.045, "30-34": 0.048, "35-39": 0.054, "40-44": 0.058, "45-49": 0.069, "50-54": 0.071, "55-59": 0.066, "60-64": 0.064, "65-69": 0.075, "70-74": 0.088, "75-79": 0.069, "80-84": 0.053, "85+": 0.049},
        "median_age": 50.8, "pct_female": 0.523, "university_rate": 0.26, "hs_rate": 0.40,
        "mean_income_jpy": 3_780_000, "marriage_rate_adj": 0.96, "mean_age_first_marriage_m": 30.0, "mean_age_first_marriage_f": 28.4,
        "tfr": 1.42, "pct_single_30_34": 0.40, "vacancy_rate": 0.183,
    },
    "Kochi": {
        "age_distribution": {"0-14": 0.098, "15-19": 0.040, "20-24": 0.040, "25-29": 0.042, "30-34": 0.046, "35-39": 0.051, "40-44": 0.056, "45-49": 0.068, "50-54": 0.071, "55-59": 0.067, "60-64": 0.066, "65-69": 0.079, "70-74": 0.092, "75-79": 0.073, "80-84": 0.056, "85+": 0.055},
        "median_age": 52.8, "pct_female": 0.527, "university_rate": 0.24, "hs_rate": 0.42,
        "mean_income_jpy": 3_580_000, "marriage_rate_adj": 0.92, "mean_age_first_marriage_m": 30.0, "mean_age_first_marriage_f": 28.4,
        "tfr": 1.41, "pct_single_30_34": 0.42, "vacancy_rate": 0.188,
    },
    "Fukuoka": {
        "age_distribution": {"0-14": 0.120, "15-19": 0.044, "20-24": 0.052, "25-29": 0.055, "30-34": 0.056, "35-39": 0.059, "40-44": 0.063, "45-49": 0.072, "50-54": 0.072, "55-59": 0.063, "60-64": 0.057, "65-69": 0.065, "70-74": 0.075, "75-79": 0.058, "80-84": 0.043, "85+": 0.046},
        "median_age": 46.8, "pct_female": 0.521, "university_rate": 0.37, "hs_rate": 0.33,
        "mean_income_jpy": 4_280_000, "marriage_rate_adj": 0.96, "mean_age_first_marriage_m": 30.4, "mean_age_first_marriage_f": 28.9,
        "tfr": 1.34, "pct_single_30_34": 0.44, "vacancy_rate": 0.127,
    },
    "Saga": {
        "age_distribution": {"0-14": 0.127, "15-19": 0.047, "20-24": 0.044, "25-29": 0.048, "30-34": 0.052, "35-39": 0.056, "40-44": 0.060, "45-49": 0.069, "50-54": 0.070, "55-59": 0.064, "60-64": 0.062, "65-69": 0.073, "70-74": 0.084, "75-79": 0.065, "80-84": 0.049, "85+": 0.030},
        "median_age": 48.0, "pct_female": 0.520, "university_rate": 0.26, "hs_rate": 0.40,
        "mean_income_jpy": 3_700_000, "marriage_rate_adj": 1.04, "mean_age_first_marriage_m": 29.7, "mean_age_first_marriage_f": 28.1,
        "tfr": 1.57, "pct_single_30_34": 0.35, "vacancy_rate": 0.143,
    },
    "Nagasaki": {
        "age_distribution": {"0-14": 0.114, "15-19": 0.044, "20-24": 0.042, "25-29": 0.044, "30-34": 0.048, "35-39": 0.053, "40-44": 0.058, "45-49": 0.069, "50-54": 0.071, "55-59": 0.066, "60-64": 0.064, "65-69": 0.076, "70-74": 0.088, "75-79": 0.069, "80-84": 0.052, "85+": 0.042},
        "median_age": 50.5, "pct_female": 0.524, "university_rate": 0.25, "hs_rate": 0.40,
        "mean_income_jpy": 3_680_000, "marriage_rate_adj": 0.96, "mean_age_first_marriage_m": 29.8, "mean_age_first_marriage_f": 28.3,
        "tfr": 1.52, "pct_single_30_34": 0.38, "vacancy_rate": 0.149,
    },
    "Kumamoto": {
        "age_distribution": {"0-14": 0.123, "15-19": 0.046, "20-24": 0.046, "25-29": 0.049, "30-34": 0.052, "35-39": 0.056, "40-44": 0.059, "45-49": 0.068, "50-54": 0.070, "55-59": 0.064, "60-64": 0.062, "65-69": 0.073, "70-74": 0.085, "75-79": 0.067, "80-84": 0.050, "85+": 0.030},
        "median_age": 48.5, "pct_female": 0.522, "university_rate": 0.28, "hs_rate": 0.38,
        "mean_income_jpy": 3_760_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 29.9, "mean_age_first_marriage_f": 28.3,
        "tfr": 1.56, "pct_single_30_34": 0.37, "vacancy_rate": 0.139,
    },
    "Oita": {
        "age_distribution": {"0-14": 0.112, "15-19": 0.043, "20-24": 0.044, "25-29": 0.046, "30-34": 0.049, "35-39": 0.054, "40-44": 0.058, "45-49": 0.069, "50-54": 0.071, "55-59": 0.066, "60-64": 0.064, "65-69": 0.076, "70-74": 0.088, "75-79": 0.069, "80-84": 0.052, "85+": 0.039},
        "median_age": 50.5, "pct_female": 0.521, "university_rate": 0.27, "hs_rate": 0.39,
        "mean_income_jpy": 3_760_000, "marriage_rate_adj": 0.98, "mean_age_first_marriage_m": 29.9, "mean_age_first_marriage_f": 28.4,
        "tfr": 1.47, "pct_single_30_34": 0.40, "vacancy_rate": 0.164,
    },
    "Miyazaki": {
        "age_distribution": {"0-14": 0.120, "15-19": 0.046, "20-24": 0.043, "25-29": 0.046, "30-34": 0.050, "35-39": 0.054, "40-44": 0.058, "45-49": 0.068, "50-54": 0.070, "55-59": 0.065, "60-64": 0.064, "65-69": 0.076, "70-74": 0.088, "75-79": 0.068, "80-84": 0.052, "85+": 0.032},
        "median_age": 50.0, "pct_female": 0.527, "university_rate": 0.24, "hs_rate": 0.42,
        "mean_income_jpy": 3_520_000, "marriage_rate_adj": 0.98, "mean_age_first_marriage_m": 29.7, "mean_age_first_marriage_f": 28.2,
        "tfr": 1.63, "pct_single_30_34": 0.38, "vacancy_rate": 0.155,
    },
    "Kagoshima": {
        "age_distribution": {"0-14": 0.122, "15-19": 0.046, "20-24": 0.044, "25-29": 0.047, "30-34": 0.050, "35-39": 0.054, "40-44": 0.058, "45-49": 0.068, "50-54": 0.070, "55-59": 0.065, "60-64": 0.064, "65-69": 0.077, "70-74": 0.088, "75-79": 0.069, "80-84": 0.052, "85+": 0.026},
        "median_age": 50.0, "pct_female": 0.527, "university_rate": 0.25, "hs_rate": 0.40,
        "mean_income_jpy": 3_600_000, "marriage_rate_adj": 0.98, "mean_age_first_marriage_m": 29.8, "mean_age_first_marriage_f": 28.3,
        "tfr": 1.60, "pct_single_30_34": 0.37, "vacancy_rate": 0.156,
    },
    "Okinawa": {
        "age_distribution": {"0-14": 0.163, "15-19": 0.054, "20-24": 0.056, "25-29": 0.057, "30-34": 0.060, "35-39": 0.062, "40-44": 0.063, "45-49": 0.072, "50-54": 0.069, "55-59": 0.059, "60-64": 0.054, "65-69": 0.061, "70-74": 0.065, "75-79": 0.044, "80-84": 0.032, "85+": 0.029},
        "median_age": 42.9, "pct_female": 0.512, "university_rate": 0.30, "hs_rate": 0.38,
        "mean_income_jpy": 3_380_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 29.8, "mean_age_first_marriage_f": 28.4,
        "tfr": 1.82, "pct_single_30_34": 0.44, "vacancy_rate": 0.119,
    },
}

NATIONAL_DEFAULTS = {
    "age_distribution": {"0-14": 0.116, "15-19": 0.042, "20-24": 0.046, "25-29": 0.050, "30-34": 0.052, "35-39": 0.056, "40-44": 0.060, "45-49": 0.068, "50-54": 0.072, "55-59": 0.065, "60-64": 0.060, "65-69": 0.068, "70-74": 0.080, "75-79": 0.065, "80-84": 0.048, "85+": 0.052},
    "median_age": 48.6, "pct_female": 0.514, "university_rate": 0.34, "hs_rate": 0.35,
    "mean_income_jpy": 4_580_000, "marriage_rate_adj": 1.00, "mean_age_first_marriage_m": 31.0, "mean_age_first_marriage_f": 29.4,
    "tfr": 1.20, "pct_single_30_34": 0.44, "vacancy_rate": 0.137,
}


def get_prefecture_profile(prefecture_name: str) -> dict:
    """Look up demographic profile for a prefecture, falling back to national defaults."""
    return PREFECTURE_DEMOGRAPHICS.get(prefecture_name, NATIONAL_DEFAULTS)
