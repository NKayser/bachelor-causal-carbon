import spacy
from tqdm import tqdm
import json

# load the best model from training
nlp = spacy.load("../../../textcat/models/cval_2/model-best")

spacy.prefer_gpu(0)

with open("../../spancat/results/textcat_all_with_pretrained_ner_output_2.jsonl", 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

with open("../results/textcat_all_with_ner_and_predictions.jsonl", 'w', encoding='utf-8') as out_file:
    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        doc = nlp(result["text"])
        result["textcat_prediction"] = doc.cats["positive"]
        out_file.write(json.dumps(result) + "\n")