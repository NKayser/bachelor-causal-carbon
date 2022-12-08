import spacy
from tqdm import tqdm
import json

from MyCode.scripts.consts import INPUT_PATH
from MyCode.scripts.utils import read_input_file

# load the best model from training
nlp = spacy.load("../textcat/models/cval_3/model-best")

spacy.prefer_gpu(0)

json_list = read_input_file()

with open(INPUT_PATH + "2", 'w', encoding='utf-8') as out_file:
    for obj in tqdm(json_list):
        doc = nlp(obj["text"])
        obj["textcat_prediction"] = doc.cats["positive"]
        out_file.write(json.dumps(obj) + "\n")