import random

import spacy
from tqdm import tqdm
from spacy.tokens import DocBin
import json


spacy.require_gpu()
nlp = spacy.blank("en")
random.seed(0)
splits = [0.7, 0.8]
include_neg_prob = 0.2
fname = "../assets/textcat_all.jsonl"

with open(fname, 'r', encoding="utf8") as json_file:
    json_list = list(json_file)

# create DocBin objects
train_db = DocBin()
dev_db = DocBin()
test_db = DocBin()

# keep track of number of positive/negative articles
distribution = [0, 0, 0, 0, 0, 0]

for json_str in tqdm(json_list):
    result = json.loads(json_str)
    doc = nlp(result["text"])
    ran1 = random.random()
    ran2 = random.random()
    labels = result["label"]
    assert len(labels) <= 1
    if len(labels) == 0:
        # not labeled
        continue
    label = labels[0]

    # exclude unsure and unlabeled data (for now)
    if label == "positive":
        doc.cats["positive"] = True
        doc.cats["negative"] = False
    elif label == "negative":
        # less negative examples to improve scores on positive texts
        if ran2 < include_neg_prob:
            doc.cats["positive"] = False
            doc.cats["negative"] = True
        else:
            continue
    else:
        # unsure
        continue

    # manual split
    if ran1 < splits[0]:
        train_db.add(doc)
        if label == "positive":
            distribution[0] += 1
        else:
            distribution[1] += 1
    elif ran1 > splits[1]:
        test_db.add(doc)
        if label == "positive":
            distribution[2] += 1
        else:
            distribution[3] += 1
    else:
        dev_db.add(doc)
        if label == "positive":
            distribution[4] += 1
        else:
            distribution[5] += 1

# save the docbin objects
train_db.to_disk(f"../corpus/textcat_train.spacy")
dev_db.to_disk(f"../corpus/textcat_dev.spacy")
test_db.to_disk(f"../corpus/textcat_test.spacy")

# print distribution
print("Train:\t" + str(distribution[0]) + " pos, " + str(distribution[1]) + " neg")
print("Dev:\t" + str(distribution[2]) + " pos, " + str(distribution[3]) + " neg")
print("Test:\t" + str(distribution[4]) + " pos, " + str(distribution[5]) + " neg")

# Train:	89 pos, 122 neg
# Dev:	    30 pos, 34 neg
# Test:	    11 pos, 22 neg

# same results after rerunning it -> random seed works
# Test set too small?