INPUT_PATH = "data/labels_and_predictions.jsonl"
TEXTCAT_MODEL_PATH = "textcat/models/cval_2/model-best"
SPANCAT_MODEL_PATH = "sententence_categorization/models/model-best"
ENT_MODEL_PATH = "entity_categorization/models-binary/model-best"
PRETRAINED_NER_MODEL = "en_core_web_trf"
TEXTCAT_THRESHOLD = 0.5

TECHNOLOGY_CATEGORIES = {
            'DRI': ['DRI', 'direct reduction', 'iron ore'],
            'EAF': ['EAF', 'Electric Arc Furnace'],
            'solar': ['solar'],
            'wind': ['wind', 'park', 'offshore'],
            'hydrogen': ['hydrogen'],
            'renewable': ['renewable energy', 'renewables', 'wind', 'solar', 'hydropower'],
            'CCGT': ['CCGT'],
            'TRT': ['TRT'],
            'HBI': ['HBI'],
            'carbon capture': ['CCUS', 'CCS', 'CCU', 'carbon capture', 'storage', 'utilization', 'store', 're-use'],
            'decommission coal plant': ['Shut down', 'shut down', 'decommission', 'coal'], # 'coal' only in combination relevant
            'plant efficiency': ['modern', 'Modern', 'heat', 'efficient', 'efficiency', 'plant', 'energy-saving',
                                 'recovery', 'production process'], # 'recovery' in combination with 'heat'
            'dust filter': [' dust', 'filter', 'sinter', 'Nm', 'smoke'], # 'industrial' not 'dust'
            'R&D': ['R&D', 'research and development', 'research'],
            'alternative fuel': ['alternative fuel', 'bio-fuel'],
            'recycling': ['recycling', 'recycle', 'recyclability', 're-use', 'scrap', 'by-product', 'waste'],
            'steel': ['steel', 'blast furnace', 'coke'],
            'cement': ['cement'],
            'microbe': ['microbe', 'microbial', 'bioethanol'],
            'general': ['environment', 'sustain', 'net zero', 'net-zero', 'neutral'],
        }

ALL_STANDARD_ENTITY_TYPES = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
                             "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]

CURRENCIES = "(C\$|EUR|Eur|eur|euro|Euros|euros|Euro|€|\u20ac|USD|Usd|usd|\$|US\$|\$US|dollar|US|U\.S\.\$|us\$|Us\$|US dollars|CAN|CAN\$|CAD|CAD\$|cad|cad\$|can|can\$|Cad|Cad\$|Can|Can\$|CHF|Chf|chf|Swiss francs|Swiss Francs|francs|franc|PLN|pln|Pln|zlotys|zloty|zlotych|\u00a3|GBP|gbp|Gbp|pounds|pound|INR|CZK|Rs\.|AUD|THB|KRW|CNY|LFL|LfL|MYR|LTL|SEK|RMB|ZAR|R|yuan|)"
MAGNITUDES = "(mn|mm|mio|mil|mln|million|m|Mn|Mio|Mil|Mln|Million|M|b|bn|billion|B|Bn|Billion|thousand|Thousand| )?"
MONEY_PATTERNS = [CURRENCIES + " ?~?(\d+([\.,]?\d*)*)[-–]?(\d*([\.,]?\d*)*)\+? ?" + MAGNITUDES,
                       "(\d+([\.,]?\d*)*)[-–]?(\d*([\.,]?\d*)*)\+? ?" + MAGNITUDES + "\.? ?" + CURRENCIES]
IGNORE_MONEY_PATTERNS = ["2 can", "\+[\d ?\-?\-?]*", "19 Eur"]

WEIGHTED_SENT_KEYWORDS = [("investment", 4),
                          ("invest", 3),
                          ("project", 2),
                          ("technology", 2),
                          ("plant", 2),
                          ("CO2", 2),
                          ("carbon", 2),
                          ("Carbon", 2),
                          ("environment", 1),
                          ("sustain", 1)] # with weights

CURRENCY_DICT = {"eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR",
                 "usd": "USD", "$": "USD", "us$": "USD",
                 "u.s.$": "USD", "us dollars": "USD", "us dollar": "USD", "us": "USD", "$us": "USD",
                 "dollar": "USD",
                 "cad": "CAD", "can$": "CAD", "can": "CAD", "cad$": "CAD", "c$": "CAD",
                 "chf": "CHF", "swiss francs": "CHF", "swiss franc": "CHF", "francs": "CHF", "franc": "CHF",
                 "pln": "PLN", "zloty": "PLN", "zlotych": "PLN", "zlotys": "PLN",
                 "gbp": "GBP", "\u00a3": "GBP", "pound": "GBP", "pounds": "GBP",
                 "czk": "CZK", "inr": "INR", "rupees": "INR", "rupee": "INR",
                 "rs.": "INR", "zar": "ZAR", "aud": "AUD", "thb": "THB", "krw": "KRW",
                 "cny": "CNY", "yuan": "CNY", "lfl": "LFL", "myr": "MYR", "ltl": "LTL", "sek": "SEK", "rmb": "RMB",
                 "r": "ZAR", "south african rand": "ZAR", "rand": "ZAR"}

MAGNITUDE_DICT = {"m": 1000000, "mio": 1000000, "mln": 1000000, "million": 1000000,
                  "b": 1000000000, "bn": 1000000000, "billion": 1000000000,
                  "thousand": 1000}

QUANTITY_IGNORE = ["magnitude", "barrel", "oil", "tcfe", "acre", "m2", "m3", "square", "cubic", "bar", "hectar", "°",
                   "degree", "tph", "bps", "KB", "KT", "tpa", "DWT", "pp"]

UNIT_DICT ={"TWh": ["Wh", 1000000000000],
            "GWh": ["Wh", 1000000000],
            "gigawatt hour": ["Wh", 1000000000],
            "MWh": ["MWh", 1000000],
            "megawatt hour": ["Wh", 1000000],
            "kWh": ["Wh", 1000],
            "Wh": ["Wh", 1],
            "kW": ["W", 1000],
            "kilowatt": ["W", 1000],
            "megawatt": ["W", 1000000],
            "MW": ["W", 1000000],
            "gigawatt": ["W", 1000000000],
            "GW": ["W", 1000000000],
            "kilogram": ["t", 0.001],
            "kg": ["t", 0.001],
            "kiloton": ["t", 1000],
            "megaton": ["t", 1000000],
            "Mt": ["t", 1000000],
            "ton": ["t", 1],
            "pound": ["t", 0.0004536],
            "lb": ["t", 0.0004536],
            "tpd": ["t/a", 0.0027397],
            "tpa": ["t/a", 1],
            "kilometer": ["m", 1000],
            "metre": ["m", 1],
            "meter": ["m", 1],
            "m": ["m", 1],
            "feet": ["m", 0.3],
            "foot": ["m", 0.3],
            "oil-equivalent": ["Wh", 1699000],
            "boe": ["Wh", 1699000],
            "gal": ["l", 3.78541],
            "barrel": ["l", 158.987],
            "l": ["l", 1],
            "h": ["h", 1]}

RELEVANT_QUANTITY_TYPES = ["mass", "power", "energy"]