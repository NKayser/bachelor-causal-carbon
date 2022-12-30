import json

import spacy
from spacy import displacy

# Using displacy on a doc without spans would result in a warning. We suppress them.
import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('models/model-best')

with open("assets/labels_and_predictions.jsonl", 'r') as json_file:
    json_list = list(json_file)[0:30]

texts = "\n ".join([json.loads(json_str)["text"] for json_str in json_list])

# Define a colour for our span category (default is grey)
colors = {'positive': '#00ff00',
          'negative': '#ff0000'}
# If other than the default 'sc', define the used spans-key and set colors
options={'spans_key':'sc', 'colors': colors}


doc = nlp(texts)
displacy.serve(doc, style="span", options=options)