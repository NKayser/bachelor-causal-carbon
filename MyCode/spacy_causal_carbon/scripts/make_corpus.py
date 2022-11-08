import spacy
from tqdm import tqdm
from spacy.tokens import DocBin
import json


def get_ent_with_id(ents, id):
    for ent in ents:
        if str(ent["id"]) == str(id):
            return ent
    return None


def get_relations_from_ent(rels, ent_id):
    return list(filter(lambda rel: str(rel["from_id"]) == str(ent_id), rels))


def get_to_ents_from_relations(ents, rels):
    return list(map(lambda rel: get_ent_with_id(ents, rel["to_id"]), rels))


def create_docbin(fname: str, basename: str, nlp):
    with open(fname, 'r') as json_file:
        json_list = list(json_file)

    db = DocBin()  # create a DocBin object

    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        #doc = nlp.make_doc(result["text"])  # create doc object from text
        doc = nlp(result["text"])
        doc_ents = []
        text_id = result["id"]
        entities = result["entities"]
        relations = result["relations"]
        for ent in entities:  # add character indexes
            start = ent["start_offset"]
            end = ent["end_offset"]
            label = ent["label"]
            id = ent["id"]
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:# or len(nlp(span.text)) > 6 or label != "emissions":
                #print("Skipping entity")
                continue
            #else:
                #if label == "core reference":
                    #relations = get_relations_from_ent(relations, id)
                    #to_ents = get_to_ents_from_relations(entities, relations)
                    #print(str(text_id) + ", len: " + str(len(relations)) + ", " + str(relations) + ": " + span.text)
                #doc_ents.append(span)
            if True: #label != "cause" and label != "effect":
                # get sentence of that ent
                # label this sentence as relevant
                for sent in doc.sents:
                    if span.text in sent.text:
                        sent_start = doc.text.index(sent.text)
                        sent_end = sent_start + len(sent.text)
                        sent_span = doc.char_span(sent_start, sent_end, label="sent_"+label)

                        if sent.text == sent_span.text:
                            doc_ents.append(sent_span)
                        else:
                            print("Sentence matching error")
                            print("Span: "+ span.text)
                            print("Sent: " + sent.text)
                            print("Start: " + str(sent_start))
                            print("End: " + str(sent_end))
                            print("New sent: " + str())
                            assert False
        doc.spans["sc"] = doc_ents  # span groups: might modify later to make core references one group
        db.add(doc)

    db.to_disk(f"./corpus/{basename}.spacy")  # save the docbin object


def main():
    #nlp = spacy.load("en_core_web_sm")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    print(nlp.pipeline)

    create_docbin("assets/cc_train.jsonl", 'train', nlp)
    create_docbin("assets/cc_trial.jsonl", 'dev', nlp)
    create_docbin("assets/cc_test.jsonl", 'eval', nlp)


main()
