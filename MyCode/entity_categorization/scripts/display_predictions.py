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
    return build_custom_suggester(balance=False)

nlp = spacy.load('models-binary/model-best')

positive_ids = get_positive_article_ids("../data/labels_and_predictions.jsonl")[0:10]
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


doc = nlp(texts)
displacy.serve(doc, style="span", options=options)