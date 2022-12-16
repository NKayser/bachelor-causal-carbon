import random

import spacy
from tqdm import tqdm
from spacy.tokens import DocBin

from MyCode.scripts.process_article import Article
from MyCode.scripts.utils import read_input_file, ent_is_in_sent, filter_ents

nlp = spacy.blank("en")
random.seed(1)
splits = [0.7, 0.8]
fname = "../data/labels_and_predictions.jsonl"

json_list = read_input_file(fname)

# create DocBin objects
train_db = DocBin()
dev_db = DocBin()
test_db = DocBin()

corresponding_labels = {
    "GPE": ["location"],
    "TECHWORD": ["technology", "core reference"],
    "FAC": ["technology", "core reference"],
    "PRODUCT": ["technology", "core reference"],
    "PERCENT": ["emissions"],
    "QUANTITY": ["emissions"],
    "DATE": ["timeline"],
    "MONEY": ["financial information"]
}

relevant_labels = set()
for arr in corresponding_labels.values():
    for val in arr:
        relevant_labels.add(val)
print(relevant_labels)

distribution = [[0, 0], [0, 0], [0, 0]]

for json_obj in tqdm(json_list):
    labeled_entities = json_obj["labeled_entities"]
    if len(labeled_entities) == 0:
        continue
    id = json_obj["id"]
    article = Article.from_article(id)
    article.set_money_ents()
    article.set_technology_ents()
    doc = nlp(json_obj["text"])
    doc_ents = []
    doc_dist = [0, 0]
    ran1 = random.random()

    for ent in article.doc.spans["sc"]:
        if ent.label_ not in corresponding_labels.keys():
            continue
        ent_in_labeled_ent = False
        for labeled_entity in labeled_entities:
            label = labeled_entity["label"]
            if label not in relevant_labels:
                continue
            labeled_ent = doc.char_span(labeled_entity["start_offset"], labeled_entity["end_offset"], label,
                                        alignment_mode="expand")
            if label in corresponding_labels[ent.label_]:
                if ent_is_in_sent(ent, labeled_ent):
                    #print(label, ent.label_, labeled_ent.text)
                    ent_in_labeled_ent = True
                    doc_dist[0] += 1
                    new_ent = doc.char_span(ent.start_char, ent.end_char, ent.label_ + " positive",
                                            alignment_mode="expand")
                    assert str(new_ent) != "None"
                    #print(new_ent)
                    doc_ents.append(new_ent)
        if ent_in_labeled_ent:
            continue
        if doc_dist[1] > doc_dist[0]:
            continue
        doc_dist[1] += 1
        new_ent = doc.char_span(ent.start_char, ent.end_char, ent.label_ + " negative", alignment_mode="expand")
        assert str(new_ent) != "None"
        #print(new_ent)
        doc_ents.append(new_ent)

    #print(doc_ents)
    doc.spans["sc"] = doc_ents
    #doc.ents = spacy.util.filter_spans(doc_ents)

    # manual split
    if ran1 < splits[0]:
        train_db.add(doc)
        distribution[0][0] += doc_dist[0]
        distribution[0][1] += doc_dist[1]
    elif ran1 > splits[1]:
        test_db.add(doc)
        distribution[1][0] += doc_dist[0]
        distribution[1][1] += doc_dist[1]
    else:
        dev_db.add(doc)
        distribution[2][0] += doc_dist[0]
        distribution[2][1] += doc_dist[1]

print(distribution)

# save the docbin objects
train_db.to_disk(f"../entity_categorization/corpus/train.spacy")
dev_db.to_disk(f"../entity_categorization/corpus/dev.spacy")
test_db.to_disk(f"../entity_categorization/corpus/test.spacy")
