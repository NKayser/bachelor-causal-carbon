INPUT_PATH = "../data/labels_and_predictions.jsonl"
TEXTCAT_MODEL_PATH = "../textcat/models/cval_2/model-best"
SPANCAT_MODEL_PATH = "../spancat/models/model-best"
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
        }