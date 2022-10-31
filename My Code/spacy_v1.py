#import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import json
from spacy import displacy

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

with open('./data/annotated_relations.jsonl', 'r') as json_file:
    json_list = list(json_file)

for json_str in tqdm(json_list):
    result = json.loads(json_str)
    doc = nlp.make_doc(result["text"]) # create doc object from text
    ents = []
    for ent in result["entities"]: # add character indexes
        start = ent["start_offset"]
        end = ent["end_offset"]
        label = ent["label"]
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.spans["sc"] = ents
    db.add(doc)

db.to_disk("./train.spacy") # save the docbin object

#displacy.serve(doc, style="span")