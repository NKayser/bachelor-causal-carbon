import random
from math import floor

import spacy
from tqdm import tqdm
from spacy.tokens import DocBin
import json


spacy.require_gpu()
nlp = spacy.blank("en")
random.seed(0)
include_neg_prob = 0.2
num_of_buckets = 6 # make 4 train, 1 dev, 1 test
fname = "../assets/textcat_all.jsonl"

with open(fname, 'r', encoding="utf8") as json_file:
    json_list = list(json_file)

# create DocBin objects
dbs = [DocBin() for i in range(0, num_of_buckets)]

# keep track of number of positive/negative articles
distribution = [[0, 0] for i in range(0, num_of_buckets)]

for json_str in tqdm(json_list):
    result = json.loads(json_str)
    doc = nlp(result["text"])
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

    # split into bucket 0 to 5
    bucket = floor(ran1 * num_of_buckets)

    dbs[bucket].add(doc)
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
for i in range(0, num_of_buckets):
    dbs[i].to_disk(f"../corpus/textcat_cval_{i}_test.spacy")
    val_bucket = (i + 1) % num_of_buckets
    dbs[val_bucket].to_disk(f"../corpus/textcat_cval_{i}_dev.spacy")

    train_db = DocBin()
    for j in range(2, num_of_buckets):
        to_merge = (i + j) % num_of_buckets
        train_db.merge(dbs[to_merge])

    train_db.to_disk(f"../corpus/textcat_cval_{i}_train.spacy")


# Bucket 0:	22 pos, 37 neg
# Bucket 1:	19 pos, 22 neg
# Bucket 2:	24 pos, 28 neg
# Bucket 3:	20 pos, 31 neg
# Bucket 4:	25 pos, 35 neg
# Bucket 5:	20 pos, 25 neg


# uneven sizes: could distribute to buckets non-randomly