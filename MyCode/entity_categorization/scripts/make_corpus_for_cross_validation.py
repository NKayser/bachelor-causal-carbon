import random
from math import floor

import spacy
from tqdm import tqdm
from spacy.tokens import DocBin

from MyCode.scripts.process_article import Article
from MyCode.scripts.utils import read_input_file, ent_is_in_sent, filter_ents, opposite_filter_ents


def create_corpus_crosseval(balanced=True):
    spacy.prefer_gpu(0)
    nlp = spacy.blank("en")
    random.seed(1)
    fname = "data/labels_and_predictions.jsonl"

    # create DocBin objects
    dbs = [DocBin() for i in range(0, num_of_buckets)]

    # keep track of number of positive/negative articles
    distribution = [[0, 0] for i in range(0, num_of_buckets)]

    json_list = read_input_file(path=fname)

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

    for json_obj in tqdm(json_list):
        ran1 = random.random() # decides which bucket
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

        ents = opposite_filter_ents(article.doc.spans["sc"], "TECHWORD")
        #ents = article.doc.spans["sc"]
        if len(ents) == 0:
            continue

        for ent in ents:
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
                        new_ent = doc.char_span(ent.start_char, ent.end_char, "positive",
                                                alignment_mode="expand")
                        assert str(new_ent) != "None"
                        #print(new_ent)
                        doc_ents.append(new_ent)
            if ent_in_labeled_ent:
                continue
            if balanced and doc_dist[1] > doc_dist[0]:
                continue
            doc_dist[1] += 1
            new_ent = doc.char_span(ent.start_char, ent.end_char, "negative", alignment_mode="expand")
            assert str(new_ent) != "None"
            #print(new_ent)
            doc_ents.append(new_ent)

        #print(doc_ents)
        doc.spans["sc"] = doc_ents
        #doc.ents = spacy.util.filter_spans(doc_ents)

        # split into bucket 0 to 5
        bucket = floor(ran1 * num_of_buckets)
        dbs[bucket].add(doc)

        # update distribution
        distribution[bucket][0] += doc_dist[0]
        distribution[bucket][1] += doc_dist[1]


    # print distribution
    for i in range(0, num_of_buckets):
        print("Bucket " + str(i) + ":\t" + str(distribution[i][0]) + " pos, " + str(distribution[i][1]) + " neg")
    return dbs


num_of_buckets = 6 # make 4 train, 1 dev, 1 test
dbs = create_corpus_crosseval(balanced=True)

# and merge buckets in different ways
# iterate through buckets which will become train set
# next bucket will become dev set
for i in range(0, num_of_buckets):
    dbs[i].to_disk(f"entity_categorization/corpus-cval/cval_{i}_test.spacy")
    val_bucket = (i + 1) % num_of_buckets
    dbs[val_bucket].to_disk(f"entity_categorization/corpus-cval/cval_{i}_dev.spacy")

    train_db = DocBin()
    for j in range(2, num_of_buckets):
        to_merge = (i + j) % num_of_buckets
        train_db.merge(dbs[to_merge])

    train_db.to_disk(f"entity_categorization/corpus-cval/cval_{i}_train.spacy")
