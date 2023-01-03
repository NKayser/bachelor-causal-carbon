import json

import spacy
from spacy import displacy, registry

# Using displacy on a doc without spans would result in a warning. We suppress them.
import warnings

from MyCode.entity_categorization.scripts.custom_span_suggester import build_custom_suggester
from MyCode.scripts.utils import get_positive_article_ids, read_input_file
from MyCode.scripts.process_article import Article

warnings.filterwarnings('ignore')

@registry.misc("article_all_ent_suggester.v1")
def suggester():
    return build_custom_suggester(balance=False, input_path="../data/labels_and_predictions.jsonl")

nlp = spacy.load('models-binary/model-best')

positive_ids = get_positive_article_ids("../data/labels_and_predictions.jsonl")[70:80]
json_list = read_input_file("../data/labels_and_predictions.jsonl")
texts = ""

for article_data in json_list:
    if article_data["id"] in positive_ids:
        texts += article_data["text"]

#texts = "\n ".join([json.loads(json_str)["text"] for json_str in json_list])

# Define a colour for our span category (default is grey)
colors = {'positive': '#00ff00',
          'negative': '#ff0000'}
# If other than the default 'sc', define the used spans-key and set colors
options={'spans_key':'sc', 'colors': colors}

def ent_to_token_slice(doc, ent):
    span = doc.char_span(ent.start_char, ent.end_char, alignment_mode="expand")
    return span[0].i, span[-1].i + 1

doc = nlp(texts)
spans = doc.spans["sc"]
for span, confidence in zip(spans, spans.attrs["scores"]):
    charspan = doc.char_span(span.start_char, span.end_char, span.label_)
    token_slice = ent_to_token_slice(doc, charspan)
    print(token_slice[0], token_slice[1], span.start_char, span.end_char, span.label_, confidence, span.text)
displacy.serve(doc, style="span", options=options)
