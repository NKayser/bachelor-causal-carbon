import json
import spacy
from tqdm import tqdm

spacy.prefer_gpu(0)

with open("../results/textcat_all_with_pretrained_ner_output.jsonl", 'r', encoding='utf-8') as broken_file:
    broken_json_list = list(broken_file)

with open("../../data/textcat_all.jsonl", 'r', encoding='utf-8') as healthy_file:
    healthy_json_list = list(healthy_file)

with open("../results/textcat_all_with_pretrained_ner_output_2.jsonl", 'w', encoding='utf-8') as out_file:
    for further_info, json_str in tqdm(zip(broken_json_list, healthy_json_list)):
        result = json.loads(json_str)
        entities_start = str(further_info).find("'entities': [")
        relevant_part = further_info[entities_start + len("'entities': "):]
        entities_end = relevant_part.find("]")
        relevant_part = relevant_part[:entities_end+1]
        relevant_part = relevant_part.replace("'", '"')
        result["entities"] = json.loads(relevant_part)
        out_file.write(json.dumps(result) + "\n")