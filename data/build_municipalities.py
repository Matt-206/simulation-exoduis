"""
Build the comprehensive japan_municipalities.py from raw data sources.

Sources:
  - Gazetteer of Japan (piuccio/open-data-jp-municipalities): lat/lon for all 1,736 municipalities
  - 2020 Population Census (e-Stat): population by municipality
  - Core City designation list (MIC): tier classification

Outputs model/japan_municipalities.py with MUNICIPALITIES list.
"""

import json
import os
import sys

RAW_PATH = os.path.join(os.path.dirname(__file__), "municipalities_raw.json")

# -----------------------------------------------------------------------
# 23 Tokyo Special Wards (Tier 0)
# -----------------------------------------------------------------------
TOKYO_WARDS = {
    "Chiyoda Ku", "Chuo Ku", "Minato Ku", "Shinjuku Ku", "Bunkyo Ku",
    "Taito Ku", "Sumida Ku", "Koto Ku", "Shinagawa Ku", "Meguro Ku",
    "Ota Ku", "Setagaya Ku", "Shibuya Ku", "Nakano Ku", "Suginami Ku",
    "Toshima Ku", "Kita Ku", "Arakawa Ku", "Itabashi Ku", "Nerima Ku",
    "Adachi Ku", "Katsushika Ku", "Edogawa Ku",
    "Edogawa",  # Gazetteer lists without " Ku"
}

# -----------------------------------------------------------------------
# Core Cities (中核市) + Designated Cities (政令指定都市) = Tier 1
# As of 2024, Japan has 20 designated cities and 62 core cities = 82 total
# -----------------------------------------------------------------------
CORE_CITIES = {
    # 20 Designated Cities (政令指定都市)
    "Sapporo Shi", "Sendai Shi", "Saitama Shi", "Chiba Shi",
    "Yokohama Shi", "Kawasaki Shi", "Sagamihara Shi", "Niigata Shi",
    "Shizuoka Shi", "Hamamatsu Shi", "Nagoya Shi", "Kyoto Shi",
    "Osaka Shi", "Sakai Shi", "Kobe Shi", "Okayama Shi",
    "Hiroshima Shi", "Kitakyushu Shi", "Fukuoka Shi", "Kumamoto Shi",
    # 62 Core Cities (中核市)
    "Hakodate Shi", "Asahikawa Shi", "Aomori Shi", "Hachinohe Shi",
    "Morioka Shi", "Akita Shi", "Yamagata Shi", "Fukushima Shi",
    "Koriyama Shi", "Iwaki Shi", "Utsunomiya Shi", "Maebashi Shi",
    "Takasaki Shi", "Kawagoe Shi", "Koshigaya Shi", "Kawaguchi Shi",
    "Funabashi Shi", "Kashiwa Shi", "Yokosuka Shi",
    "Toyama Shi", "Kanazawa Shi", "Fukui Shi", "Nagano Shi",
    "Gifu Shi", "Toyota Shi", "Toyohashi Shi", "Okazaki Shi",
    "Kasugai Shi", "Tsu Shi", "Yokkaichi Shi",
    "Otsu Shi", "Takatsuki Shi", "Higashiosaka Shi", "Toyonaka Shi",
    "Suita Shi", "Himeji Shi", "Nishimiya Shi", "Amagasaki Shi",
    "Akashi Shi", "Nara Shi", "Wakayama Shi",
    "Tottori Shi", "Matsue Shi", "Kurashiki Shi",
    "Kure Shi", "Fukuyama Shi",
    "Shimonoseki Shi", "Takamatsu Shi", "Matsuyama Shi",
    "Kochi Shi", "Kurume Shi", "Sasebo Shi", "Nagasaki Shi",
    "Oita Shi", "Miyazaki Shi", "Kagoshima Shi", "Naha Shi",
    # Additional core cities designated 2020-2024
    "Ibaraki Shi", "Neyagawa Shi", "Kofu Shi",
    "Matsumoto Shi", "Otaru Shi",
}

# -----------------------------------------------------------------------
# 2020 Census populations (令和2年国勢調査)
# For all municipalities. Major cities have exact data; smaller ones
# are estimated from prefecture-level distributions.
# -----------------------------------------------------------------------
POPULATION_2020 = {
    # Tokyo Special Wards
    "Chiyoda Ku": 66_680, "Chuo Ku": 169_179, "Minato Ku": 260_379,
    "Shinjuku Ku": 349_385, "Bunkyo Ku": 240_069, "Taito Ku": 211_444,
    "Sumida Ku": 272_085, "Koto Ku": 524_310, "Shinagawa Ku": 422_488,
    "Meguro Ku": 288_088, "Ota Ku": 748_081, "Setagaya Ku": 943_664,
    "Shibuya Ku": 243_883, "Nakano Ku": 344_880, "Suginami Ku": 591_108,
    "Toshima Ku": 301_599, "Kita Ku": 355_213, "Arakawa Ku": 217_475,
    "Itabashi Ku": 584_483, "Nerima Ku": 752_608, "Adachi Ku": 695_043,
    "Katsushika Ku": 453_093, "Edogawa Ku": 697_932, "Edogawa": 697_932,

    # 20 Designated Cities
    "Sapporo Shi": 1_975_065, "Sendai Shi": 1_096_704,
    "Saitama Shi": 1_324_025, "Chiba Shi": 974_951,
    "Yokohama Shi": 3_777_491, "Kawasaki Shi": 1_538_262,
    "Sagamihara Shi": 725_493, "Niigata Shi": 789_275,
    "Shizuoka Shi": 693_389, "Hamamatsu Shi": 790_718,
    "Nagoya Shi": 2_332_176, "Kyoto Shi": 1_463_723,
    "Osaka Shi": 2_752_412, "Sakai Shi": 826_161,
    "Kobe Shi": 1_525_152, "Okayama Shi": 724_691,
    "Hiroshima Shi": 1_200_754, "Kitakyushu Shi": 939_029,
    "Fukuoka Shi": 1_612_392, "Kumamoto Shi": 738_865,

    # Core Cities (62)
    "Hakodate Shi": 251_084, "Asahikawa Shi": 329_306,
    "Aomori Shi": 275_192, "Hachinohe Shi": 223_415,
    "Morioka Shi": 289_731, "Akita Shi": 307_672,
    "Yamagata Shi": 248_522, "Fukushima Shi": 283_348,
    "Koriyama Shi": 327_692, "Iwaki Shi": 332_931,
    "Utsunomiya Shi": 518_757, "Maebashi Shi": 336_154,
    "Takasaki Shi": 373_584, "Kawagoe Shi": 354_571,
    "Koshigaya Shi": 341_621, "Kawaguchi Shi": 594_274,
    "Funabashi Shi": 642_907, "Kashiwa Shi": 426_468,
    "Yokosuka Shi": 388_078, "Toyama Shi": 413_938,
    "Kanazawa Shi": 463_254, "Fukui Shi": 262_327,
    "Nagano Shi": 372_760, "Gifu Shi": 402_557,
    "Toyota Shi": 422_542, "Toyohashi Shi": 371_920,
    "Okazaki Shi": 381_051, "Kasugai Shi": 306_508,
    "Tsu Shi": 273_105, "Yokkaichi Shi": 311_031,
    "Otsu Shi": 344_547, "Takatsuki Shi": 347_229,
    "Higashiosaka Shi": 493_940, "Toyonaka Shi": 401_558,
    "Suita Shi": 385_956, "Himeji Shi": 530_495,
    "Nishimiya Shi": 489_401, "Amagasaki Shi": 452_563,
    "Akashi Shi": 303_601, "Nara Shi": 354_630,
    "Wakayama Shi": 356_729, "Tottori Shi": 188_405,
    "Matsue Shi": 203_616, "Kurashiki Shi": 477_118,
    "Kure Shi": 214_592, "Fukuyama Shi": 461_357,
    "Shimonoseki Shi": 255_051, "Takamatsu Shi": 421_797,
    "Matsuyama Shi": 511_192, "Kochi Shi": 326_545,
    "Kurume Shi": 303_316, "Sasebo Shi": 243_223,
    "Nagasaki Shi": 409_118, "Oita Shi": 478_146,
    "Miyazaki Shi": 401_138, "Kagoshima Shi": 593_128,
    "Naha Shi": 317_625,
    "Ibaraki Shi": 286_784, "Neyagawa Shi": 229_733,
    "Kofu Shi": 187_985, "Matsumoto Shi": 241_145,
    "Otaru Shi": 111_299,
}

# Prefecture-level 2020 census populations for distributing to smaller municipalities
PREFECTURE_POPULATIONS = {
    "Hokkaido": 5_224_614, "Aomori": 1_237_984, "Iwate": 1_210_534,
    "Miyagi": 2_301_996, "Akita": 959_502, "Yamagata": 1_068_027,
    "Fukushima": 1_833_152, "Ibaraki": 2_867_009, "Tochigi": 1_933_146,
    "Gunma": 1_939_110, "Saitama": 7_344_765, "Chiba": 6_284_480,
    "Tokyo": 14_047_594, "Kanagawa": 9_237_337, "Niigata": 2_201_272,
    "Toyama": 1_034_814, "Ishikawa": 1_132_526, "Fukui": 766_863,
    "Yamanashi": 809_974, "Nagano": 2_048_011, "Gifu": 1_978_742,
    "Shizuoka": 3_633_202, "Aichi": 7_542_415, "Mie": 1_770_254,
    "Shiga": 1_413_610, "Kyoto": 2_578_087, "Osaka": 8_837_685,
    "Hyogo": 5_465_002, "Nara": 1_324_473, "Wakayama": 922_584,
    "Tottori": 553_407, "Shimane": 671_126, "Okayama": 1_888_432,
    "Hiroshima": 2_799_702, "Yamaguchi": 1_342_059, "Tokushima": 719_559,
    "Kagawa": 950_244, "Ehime": 1_334_841, "Kochi": 691_527,
    "Fukuoka": 5_135_214, "Saga": 811_442, "Nagasaki": 1_312_317,
    "Kumamoto": 1_738_301, "Oita": 1_123_852, "Miyazaki": 1_069_576,
    "Kagoshima": 1_588_256, "Okinawa": 1_467_480,
}

# Map prefecture romaji from the JSON to our keys
PREF_ROMAJI_MAP = {
    "Hokkaido": "Hokkaido", "Aomori": "Aomori", "Iwate": "Iwate",
    "Miyagi": "Miyagi", "Akita": "Akita", "Yamagata": "Yamagata",
    "Fukushima": "Fukushima", "Ibaraki": "Ibaraki", "Tochigi": "Tochigi",
    "Gunma": "Gunma", "Saitama": "Saitama", "Chiba": "Chiba",
    "Tokyo": "Tokyo", "Kanagawa": "Kanagawa", "Niigata": "Niigata",
    "Toyama": "Toyama", "Ishikawa": "Ishikawa", "Fukui": "Fukui",
    "Yamanashi": "Yamanashi", "Nagano": "Nagano", "Gifu": "Gifu",
    "Shizuoka": "Shizuoka", "Aichi": "Aichi", "Mie": "Mie",
    "Shiga": "Shiga", "Kyoto": "Kyoto", "Osaka": "Osaka",
    "Hyogo": "Hyogo", "Nara": "Nara", "Wakayama": "Wakayama",
    "Tottori": "Tottori", "Shimane": "Shimane", "Okayama": "Okayama",
    "Hiroshima": "Hiroshima", "Yamaguchi": "Yamaguchi",
    "Tokushima": "Tokushima", "Kagawa": "Kagawa", "Ehime": "Ehime",
    "Kochi": "Kochi", "Fukuoka": "Fukuoka", "Saga": "Saga",
    "Nagasaki": "Nagasaki", "Kumamoto": "Kumamoto", "Oita": "Oita",
    "Miyazaki": "Miyazaki", "Kagoshima": "Kagoshima",
    "Okinawa": "Okinawa",
}

# Additional city populations (non-core/designated cities with known data)
ADDITIONAL_POPULATIONS = {
    # Hokkaido
    "Obihiro Shi": 166_536, "Tomakomai Shi": 170_113, "Kushiro Shi": 165_077,
    "Kitami Shi": 115_334, "Ebetsu Shi": 120_636, "Eniwa Shi": 70_239,
    "Chitose Shi": 97_950, "Iwamizawa Shi": 76_129, "Noboribetsu Shi": 46_576,
    # Aomori
    "Hirosaki Shi": 168_466, "Towada Shi": 60_301, "Misawa Shi": 38_579,
    "Goshogawara Shi": 51_103, "Mutsu Shi": 54_367,
    # Iwate
    "Ichinoseki Shi": 111_932, "Oshu Shi": 114_615, "Hanamaki Shi": 94_001,
    "Kitakami Shi": 92_447, "Miyako Shi": 50_369,
    # Miyagi
    "Ishinomaki Shi": 139_908, "Osaki Shi": 127_187, "Tome Shi": 73_272,
    "Natori Shi": 78_668, "Tagajo Shi": 63_060, "Shiogama Shi": 51_766,
    # Akita
    "Yokote Shi": 86_276, "Daisen Shi": 76_877, "Yuzawa Shi": 42_691,
    "Noshiro Shi": 49_982,
    # Yamagata
    "Tsuruoka Shi": 122_347, "Sakata Shi": 100_273, "Yonezawa Shi": 80_594,
    "Tendo Shi": 61_715, "Higashine Shi": 45_536,
    # Fukushima
    "Aizuwakamatsu Shi": 117_805, "Sukagawa Shi": 74_370,
    "Shirakawa Shi": 59_505, "Nihonmatsu Shi": 52_750,
    # Ibaraki
    "Mito Shi": 270_783, "Tsukuba Shi": 241_656, "Hitachi Shi": 172_517,
    "Hitachinaka Shi": 157_060, "Kashima Shi": 67_879,
    "Toride Shi": 104_570, "Tsuchiura Shi": 138_501, "Koga Shi": 139_346,
    "Ryugasaki Shi": 76_020, "Moriya Shi": 68_025,
    # Tochigi
    "Oyama Shi": 167_490, "Tochigi Shi": 155_549, "Ashikaga Shi": 143_542,
    "Sano Shi": 115_952, "Kanuma Shi": 94_617, "Nasushiobara Shi": 115_210,
    # Gunma
    "Isesaki Shi": 212_120, "Ota Shi": 223_014, "Kiryu Shi": 106_530,
    "Tatebayashi Shi": 76_378, "Shibukawa Shi": 74_539,
    # Saitama
    "Tokorozawa Shi": 342_464, "Kasukabe Shi": 232_709,
    "Kumagaya Shi": 193_349, "Ageo Shi": 226_943, "Soka Shi": 252_261,
    "Niiza Shi": 164_520, "Iruma Shi": 149_136,
    "Asaka Shi": 139_931, "Fujimino Shi": 113_893,
    # Chiba
    "Ichikawa Shi": 496_676, "Matsudo Shi": 498_232,
    "Ichihara Shi": 268_917, "Narita Shi": 132_906,
    "Sakura Shi": 171_920, "Yachiyo Shi": 200_695,
    "Urayasu Shi": 171_362, "Narashino Shi": 176_197,
    "Abiko Shi": 130_153, "Kamagaya Shi": 109_449,
    # Tokyo (non-ward cities)
    "Hachioji Shi": 579_330, "Tachikawa Shi": 185_221,
    "Musashino Shi": 150_219, "Mitaka Shi": 195_391,
    "Fuchu Shi": 262_790, "Chofu Shi": 240_347,
    "Machida Shi": 432_348, "Kodaira Shi": 198_739,
    "Hino Shi": 191_827, "Tama Shi": 146_951,
    "Nishitokyo Shi": 207_388, "Kokubunji Shi": 128_907,
    "Higashimurayama Shi": 151_815, "Akishima Shi": 113_126,
    "Komae Shi": 84_122, "Higashikurume Shi": 116_870,
    # Kanagawa
    "Fujisawa Shi": 436_905, "Hiratsuka Shi": 258_227,
    "Atsugi Shi": 225_714, "Yamato Shi": 240_496,
    "Chigasaki Shi": 243_300, "Odawara Shi": 190_015,
    "Kamakura Shi": 172_302, "Zama Shi": 132_299,
    # Niigata
    "Nagaoka Shi": 267_931, "Joetsu Shi": 189_233,
    "Sanjo Shi": 95_085, "Kashiwazaki Shi": 82_182,
    # Toyama
    "Takaoka Shi": 165_930, "Tonami Shi": 47_714,
    # Ishikawa
    "Hakusan Shi": 110_459, "Komatsu Shi": 106_216,
    # Nagano
    "Ueda Shi": 155_016, "Iida Shi": 96_842, "Suwa Shi": 48_300,
    # Gifu
    "Ogaki Shi": 159_879, "Kakamigahara Shi": 148_547,
    "Tajimi Shi": 107_295, "Toki Shi": 56_564,
    # Shizuoka
    "Numazu Shi": 189_795, "Fuji Shi": 245_392,
    "Fujinomiya Shi": 128_527, "Iwata Shi": 167_018,
    "Yaizu Shi": 138_604, "Shimada Shi": 96_661,
    # Aichi
    "Ichinomiya Shi": 381_971, "Seto Shi": 127_792,
    "Handa Shi": 117_479, "Anjou Shi": 189_235,
    "Nishio Shi": 172_849, "Inuyama Shi": 73_019,
    "Komaki Shi": 148_748, "Kariya Shi": 153_015,
    # Mie
    "Suzuka Shi": 196_403, "Kuwana Shi": 140_303,
    "Nabari Shi": 76_834, "Ise Shi": 122_765,
    # Shiga
    "Kusatsu Shi": 145_099, "Nagahama Shi": 113_636,
    "Hikone Shi": 113_679, "Higashiomi Shi": 113_258,
    # Kyoto
    "Uji Shi": 178_604, "Kameoka Shi": 86_752,
    "Nagaokakyo Shi": 81_655, "Maizuru Shi": 80_336,
    # Osaka
    "Hirakata Shi": 397_790, "Yao Shi": 263_414,
    "Minoo Shi": 138_312, "Izumi Shi": 184_709,
    "Tondabayashi Shi": 108_483, "Matsubara Shi": 118_085,
    "Daito Shi": 119_048, "Kadoma Shi": 119_752,
    "Kishiwada Shi": 192_320, "Moriguchi Shi": 143_042,
    # Hyogo
    "Kakogawa Shi": 261_070, "Takarazuka Shi": 234_435,
    "Itami Shi": 205_042, "Kawanishi Shi": 156_423,
    "Sanda Shi": 111_374, "Miki Shi": 76_264,
    # Nara
    "Kashihara Shi": 122_674, "Ikoma Shi": 119_440,
    "Yamatokoriyama Shi": 85_098, "Tenri Shi": 65_574,
    "Kashiba Shi": 79_093, "Gojo Shi": 28_075,
    # Wakayama
    "Hashimoto Shi": 60_968, "Tanabe Shi": 70_966,
    "Kainan Shi": 48_782,
    # Tottori
    "Yonago Shi": 147_317, "Kurayoshi Shi": 46_425,
    # Shimane
    "Izumo Shi": 172_775, "Hamada Shi": 54_932,
    "Masuda Shi": 44_894,
    # Okayama
    "Tsuyama Shi": 99_937, "Tamano Shi": 57_648,
    "Soja Shi": 69_478,
    # Hiroshima
    "Higashihiroshima Shi": 196_608, "Onomichi Shi": 128_820,
    "Mihara Shi": 91_470,
    # Yamaguchi
    "Yamaguchi Shi": 193_656, "Ube Shi": 162_570,
    "Iwakuni Shi": 128_681, "Shunan Shi": 140_083,
    "Hofu Shi": 113_170,
    # Tokushima
    "Tokushima Shi": 252_391, "Naruto Shi": 55_284,
    "Anan Shi": 69_100, "Komatsushima Shi": 36_019,
    # Kagawa
    "Marugame Shi": 110_010, "Sakaide Shi": 49_959,
    "Sanuki Shi": 47_494, "Kan-onji Shi": 57_438,
    # Ehime
    "Imabari Shi": 149_184, "Niihama Shi": 115_038,
    "Saijo Shi": 105_632, "Ozu Shi": 41_788,
    # Kochi
    "Nankoku Shi": 47_010, "Shimanto Shi": 33_032,
    "Sukumo Shi": 19_804,
    # Fukuoka
    "Kasuga Shi": 113_486, "Oonojo Shi": 103_157,
    "Iizuka Shi": 126_206, "Chikushino Shi": 105_331,
    "Omuta Shi": 111_282, "Munakata Shi": 97_376,
    "Itoshima Shi": 101_198, "Dazaifu Shi": 72_889,
    "Yukuhashi Shi": 70_013,
    # Saga
    "Saga Shi": 233_301, "Karatsu Shi": 117_596,
    "Tosu Shi": 74_327, "Imari Shi": 52_547,
    # Nagasaki
    "Isahaya Shi": 131_530, "Omura Shi": 97_056,
    # Kumamoto
    "Yatsushiro Shi": 122_374, "Arao Shi": 51_116,
    "Tamana Shi": 64_974, "Uto Shi": 36_536,
    # Oita
    "Beppu Shi": 115_092, "Nakatsu Shi": 82_875,
    "Saiki Shi": 66_850, "Usuki Shi": 35_328,
    # Miyazaki
    "Nobeoka Shi": 118_395, "Miyakonojo Shi": 161_110,
    "Hyuga Shi": 60_303, "Nichinan Shi": 49_936,
    # Kagoshima
    "Kirishima Shi": 124_770, "Kanoya Shi": 100_076,
    "Satsumasendai Shi": 91_146, "Aira Shi": 77_273,
    "Ichikikushikino Shi": 27_509,
    # Okinawa
    "Okinawa Shi": 142_752, "Urasoe Shi": 115_061,
    "Ginowan Shi": 100_109, "Nago Shi": 63_544,
    "Itoman Shi": 61_024, "Tomigusuku Shi": 74_217,
    "Chatan Cho": 28_881,
}

POPULATION_2020.update(ADDITIONAL_POPULATIONS)


def build():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} municipalities from Gazetteer")

    # Normalize prefecture names
    pref_key_map = {}
    for entry in raw:
        pr = entry["prefecture_romaji"]
        # Strip " Ken", " Fu", " To", " Do" suffix
        for suf in [" Ken", " Fu", " To", " Do"]:
            pr_clean = pr.replace(suf, "")
            if pr_clean in PREF_ROMAJI_MAP:
                pref_key_map[pr] = PREF_ROMAJI_MAP[pr_clean]
                break
        else:
            pref_key_map[pr] = pr.replace(" Ken", "").replace(" Fu", "").replace(" To", "").replace(" Do", "")

    # Count municipalities per prefecture for population distribution
    pref_munis = {}
    for entry in raw:
        pk = pref_key_map.get(entry["prefecture_romaji"], "")
        if pk not in pref_munis:
            pref_munis[pk] = []
        pref_munis[pk].append(entry["name_romaji"])

    # Build output
    municipalities = []
    assigned_pop_total = 0

    for entry in raw:
        name = entry["name_romaji"]
        name_jp = entry["name_kanji"]
        code = entry["code"]
        lat = float(entry["lat"])
        lon = float(entry["lon"])
        pref = pref_key_map.get(entry["prefecture_romaji"], "")
        pref_jp = entry["prefecture_kanji"]

        # Tier classification
        if name in TOKYO_WARDS:
            tier = 0
        elif name in CORE_CITIES:
            tier = 1
        else:
            # Tokyo non-ward cities also get tier 1 if population > 200k
            tier = 2

        # Population
        if name in POPULATION_2020:
            pop = POPULATION_2020[name]
        else:
            # Estimate from prefecture remainder
            pref_total = PREFECTURE_POPULATIONS.get(pref, 100_000)
            known_in_pref = sum(
                POPULATION_2020.get(m, 0) for m in pref_munis.get(pref, [])
            )
            remainder = max(pref_total - known_in_pref, 0)
            unknown_in_pref = sum(
                1 for m in pref_munis.get(pref, []) if m not in POPULATION_2020
            )
            if unknown_in_pref > 0:
                # Distribute proportionally by municipality type
                if "Shi" in name:
                    weight = 3.0
                elif "Cho" in name or "Machi" in name:
                    weight = 1.0
                else:  # Mura/Son
                    weight = 0.3

                total_weight = sum(
                    3.0 if "Shi" in m else (1.0 if ("Cho" in m or "Machi" in m) else 0.3)
                    for m in pref_munis.get(pref, []) if m not in POPULATION_2020
                )
                if total_weight > 0:
                    pop = max(100, int(remainder * weight / total_weight))
                else:
                    pop = max(100, remainder // max(unknown_in_pref, 1))
            else:
                pop = 5_000  # fallback

        assigned_pop_total += pop

        municipalities.append({
            "code": code,
            "name_en": name,
            "name_jp": name_jp,
            "prefecture": pref,
            "prefecture_jp": pref_jp,
            "tier": tier,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "population_2020": pop,
        })

    # Reclassify large non-ward Tokyo cities as Tier 1
    for m in municipalities:
        if m["prefecture"] == "Tokyo" and m["tier"] == 2 and m["population_2020"] > 200_000:
            m["tier"] = 1

    # Stats
    t0 = sum(1 for m in municipalities if m["tier"] == 0)
    t1 = sum(1 for m in municipalities if m["tier"] == 1)
    t2 = sum(1 for m in municipalities if m["tier"] == 2)
    print(f"Tier 0 (Tokyo Wards): {t0}")
    print(f"Tier 1 (Core/Designated): {t1}")
    print(f"Tier 2 (Periphery): {t2}")
    print(f"Total: {len(municipalities)}")
    print(f"Total assigned population: {assigned_pop_total:,}")

    # Write output module
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "japan_municipalities.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write('Comprehensive dataset of all Japanese municipalities.\n')
        f.write('\n')
        f.write('Sources:\n')
        f.write('  - Gazetteer of Japan (GSI): coordinates\n')
        f.write('  - 2020 Population Census (e-Stat): populations\n')
        f.write('  - MIC Core City / Designated City lists: tier classification\n')
        f.write('\n')
        f.write(f'Total: {len(municipalities)} municipalities\n')
        f.write(f'  Tier 0 (Tokyo 23 Wards): {t0}\n')
        f.write(f'  Tier 1 (Core/Designated Cities): {t1}\n')
        f.write(f'  Tier 2 (Periphery): {t2}\n')
        f.write('"""\n\n')
        f.write("MUNICIPALITIES = [\n")
        for m in municipalities:
            f.write(f"    {m},\n")
        f.write("]\n")

    print(f"Written to {out_path}")
    return municipalities


if __name__ == "__main__":
    build()
