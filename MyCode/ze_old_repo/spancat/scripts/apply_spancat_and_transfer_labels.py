import spacy
from tqdm import tqdm
import json

# load the best model from training
nlp = spacy.load("../../../sententence_categorization/models/model-best")

spacy.prefer_gpu(0)

with open("../../textcat/results/textcat_all_with_ner_and_predictions.jsonl", 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

with open("../../data/annotated_relations.jsonl", 'r', encoding='utf-8') as json_file:
    annotated_list = list(json_file)

with open("../results/textcat_all_with_ner_span_labels_and_predictions.jsonl", 'w', encoding='utf-8') as out_file:
    running_span_id = 0
    found_annotated = 0
    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        article_title = result["title"]
        result["labeled_entities"] = []
        for annotated in annotated_list:
            annotated_obj = json.loads(annotated)
            if annotated_obj["title"] == article_title:
                found_annotated += 1
                result["labeled_entities"] = annotated_obj["entities"]
                result["relations"] = annotated_obj["relations"]
        doc = nlp(result["text"])
        result["predicted_sent_spans"] = []

        for span in doc.spans['sc']:
            result["predicted_sent_spans"].append(
                {"id": running_span_id, "label": span.label_, "start_offset": span.start_char, "end_offset": span.end_char})
            running_span_id += 1

        out_file.write(json.dumps(result) + "\n")

    print(found_annotated)