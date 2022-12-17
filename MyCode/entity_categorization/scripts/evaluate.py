import spacy
from spacy.tokens import DocBin

from MyCode.entity_categorization.scripts.make_corpus import create_corpus_docbins
from MyCode.scripts.process_article import Article
from MyCode.scripts.utils import write_json_to_file

docbin = DocBin().from_disk("entity_categorization/corpus/test.spacy")
nlp = spacy.load("entity_categorization/models/model-best")

labels = ["DATE", "FAC", "GPE", "PRODUCT", "MONEY", "PERCENT", "QUANTITY"]
pos_labels = [label + " positive" for label in labels]
neg_labels = [label + " negative" for label in labels]
class_labels = pos_labels + neg_labels

confusion_matrix = {actual_class: {predicted_class: 0
                                   for predicted_class in class_labels}
                    for actual_class in class_labels}

#train_db, dev_db, test_db = create_corpus_docbins()

for doc in list(docbin.get_docs(nlp.vocab)):
    predicted_doc = nlp(doc.text)
    #print([[span.start_char, span.end_char, span.text, span.label_] for span in predicted_doc.ents])
    print(predicted_doc["sc"])
    for predicted_span in predicted_doc.ents:
        for true_span in doc.spans["sc"]:
            if predicted_span.start_char == true_span.start_char and predicted_span.end_char == true_span.end_char:
                #true_label = true_span.label_.split()
                #predicted_label = predicted_span.split()
                confusion_matrix[true_span.label_][predicted_span.label_] += 1
                break

filtered_confusion_matrix = {k: {k2: v2 for k2, v2 in v.items() if v2 > 0} for k, v in confusion_matrix.items()}
print(filtered_confusion_matrix)

write_json_to_file(filtered_confusion_matrix, "entity_categorization/metrics/confusion_matrix.json")