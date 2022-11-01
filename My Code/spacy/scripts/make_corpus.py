import spacy
from tqdm import tqdm
from spacy.tokens import DocBin
import json

def create_docbin(fname: str, basename: str, nlp):
    with open(fname, 'r') as json_file:
        json_list = list(json_file)

    db = DocBin() # create a DocBin object

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
        doc.spans["sc"] = ents # span groups: might modify later to make core references one group
        db.add(doc)

    db.to_disk(f"./corpus/{basename}.spacy") # save the docbin object

def main():
    nlp = spacy.load("en_core_web_sm")

    create_docbin("assets/cc_train.jsonl", 'train', nlp)
    create_docbin("assets/cc_trial.jsonl", 'dev', nlp)
    create_docbin("assets/cc_test.jsonl", 'eval', nlp)