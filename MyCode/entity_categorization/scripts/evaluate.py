import json

import spacy
from spacy import registry
from spacy.tokens import DocBin
from tqdm import tqdm

from MyCode.entity_categorization.scripts.custom_span_suggester import build_custom_suggester
from MyCode.scripts.utils import write_json_to_file


@registry.misc("article_all_ent_suggester.v1")
def suggester():
    return build_custom_suggester()

#db_all = DocBin()
#db_train = DocBin().from_disk("entity_categorization/corpus/train.spacy")
#db_dev = DocBin().from_disk("entity_categorization/corpus/dev.spacy")
db_test = DocBin().from_disk("corpus/test.spacy")
#db_all.merge(db_dev)
#db_all.merge(db_test)
nlp = spacy.load("models-binary/model-best")

labels = ["DATE", "FAC", "GPE", "PRODUCT", "MONEY", "PERCENT", "QUANTITY"]
#class_labels = []
#for label in labels:
#    class_labels.append(label + " positive")
#    class_labels.append(label + " negative")
class_labels = ["positive", "negative"]

confusion_matrix = {actual_class: {predicted_class: 0
                                   for predicted_class in class_labels}
                    for actual_class in class_labels}

for doc in tqdm(list(db_test.get_docs(nlp.vocab))):
    assert doc is not None
    assert doc.text is not None
    try:
        predicted_doc = nlp(doc.text)
    except ValueError:
        print("Error with loading doc")
        continue
    #print([[span.start_char, span.end_char, span.text, span.label_] for span in predicted_doc.spans["sc"]])
    for predicted_span in predicted_doc.spans["sc"]:
        for true_span in doc.spans["sc"]:
            if predicted_span.start_char == true_span.start_char and predicted_span.end_char == true_span.end_char:
                found_predicted = True
                #true_label = true_span.label_.split()
                #predicted_label = predicted_span.split()
                confusion_matrix[true_span.label_][predicted_span.label_] += 1
                break


filtered_confusion_matrix = {k: {k2: v2 for k2, v2 in v.items() if v2 > 0} for k, v in confusion_matrix.items()}
print(json.dumps(filtered_confusion_matrix, indent=4))

write_json_to_file(filtered_confusion_matrix, "metrics/binary_confusion_matrix.json")