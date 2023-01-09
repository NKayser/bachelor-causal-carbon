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