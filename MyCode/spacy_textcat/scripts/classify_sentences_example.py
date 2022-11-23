import json

import spacy
from tqdm import tqdm

# load the best model from training
nlp = spacy.load("../models/model-best")
nlp.add_pipe('sentencizer')
spacy.require_gpu(0)

with open("../assets/textcat_all.jsonl", 'r', encoding="utf8") as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    doc = nlp(result["text"])

    #print("Doc id " + str(result["id"]) + ": '" + result["title"] + "'")
    sent_num = 1
    if doc.cats["positive"] > 0.5:
        doc_class = "positive"
    else:
        doc_class = "negative"

    for sentence in doc.sents:
        s_doc = nlp(sentence.text)
        val = s_doc.cats['positive']
        if val > 0.5:
            print("Doc " + str(result["id"]) + " (" + doc_class + "), sent " + str(sent_num) + ", val %.2f" % val + " -> positive: " + sentence.text)
        else:
            if val > 0.1:
                # print anyway, because slightly interesting
                print("Doc " + str(result["id"]) + " (" + doc_class + "), sent " + str(sent_num) + ", val %.2f" % val + " -> negative: " + sentence.text)
        sent_num += 1