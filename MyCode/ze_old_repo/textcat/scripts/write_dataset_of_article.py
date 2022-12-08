import random
from math import floor

import spacy
from tqdm import tqdm
import json


nlp = spacy.blank("en")
random.seed(0)
include_neg_prob = 0.2
num_of_buckets = 6 # make 4 train, 1 dev, 1 test
fname = "../data/labels_and_predictions.jsonl"

with open(fname, 'r', encoding="utf8") as json_file:
    json_list = list(json_file)


# keep track of number of positive/negative articles
distribution = [[0, 0] for i in range(0, num_of_buckets)]
buckets = [[] for i in range(0, num_of_buckets)]

for json_str in tqdm(json_list):
    result = json.loads(json_str)
    ran1 = random.random() # decides which bucket
    ran2 = random.random() # decides if to include negative sample
    labels = result["label"]
    assert len(labels) <= 1
    if len(labels) == 0:
        # not labeled
        continue
    label = labels[0]

    # exclude unsure and unlabeled data (for now)
    if label == "positive":
        None
    elif label == "negative":
        # less negative examples to improve scores on positive texts
        if ran2 < include_neg_prob:
            None
        else:
            continue
    else:
        # unsure
        continue

    # split into bucket 0 to 5
    bucket = floor(ran1 * num_of_buckets)

    buckets[bucket].append(result["id"])
    if label == "positive":
        distribution[bucket][0] += 1
    else:
        distribution[bucket][1] += 1

# print distribution
for i in range(0, num_of_buckets):
    print("Bucket " + str(i) + ":\t" + str(distribution[i][0]) + " pos, " + str(distribution[i][1]) + " neg")

# and merge buckets in different ways
# iterate through buckets which will become train set
# next bucket will become dev set
with open(fname + "2", 'w', encoding="utf8") as json_file:
    i = 3 # choose third cval set

    for json_str in tqdm(json_list):
        input_line = json.loads(json_str)
        doc_id = input_line["id"]

        if doc_id in buckets[i]:
            # search for doc with id and write "test"
            input_line["dataset"] = "test"
            json_file.write(json.dumps(input_line) + "\n")
            continue
        val_bucket = (i + 1) % num_of_buckets
        if doc_id in buckets[val_bucket]:
            # search for doc with id and write "test"
            input_line["dataset"] = "validation"
            json_file.write(json.dumps(input_line) + "\n")
            continue
        for j in range(2, num_of_buckets):
            to_merge = (i + j) % num_of_buckets
            if doc_id in buckets[to_merge]:
                # search for doc with id and write "train"
                input_line["dataset"] = "train"
                json_file.write(json.dumps(input_line) + "\n")
                continue

        input_line["dataset"] = None
        json_file.write(json.dumps(input_line) + "\n")