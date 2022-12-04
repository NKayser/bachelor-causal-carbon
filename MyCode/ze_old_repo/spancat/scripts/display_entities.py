import json
import random

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_trf")
spacy.prefer_gpu(0)

with open("../../data/textcat_only_positive.jsonl", 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

    text = " ".join([json.loads(json_str)["text"] for json_str in json_list])

    doc = nlp(text)

    entities = ['NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'DATE', 'PERCENT',
                'MONEY', 'QUANTITY']
    colors = {"ENT": "#E8DAEF"}

    def color_generator(number_of_colors):
        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
        return color


    color = color_generator(len(entities))
    for i in range(len(entities)):
        colors[entities[i]] = color[i]

    # If other than the default 'sc', define the used spans-key and set colors
    options = {'colors': colors}

    displacy.serve(doc, style="ent", options=options)