# bachelor-causal-carbon

To run spacy project:

* cd to MyCode/spacy-causal-carbon
* pip install -r requirements.txt
* python -m spacy project assets
* python -m spacy project run corpus
* When changing base_config: python -m spacy init fill-config configs/base_config.cfg configs/config.cfg
* python -m spacy project run train
* python -m spacy project run evaluate

There is also a display_spans script to get a better idea of how the spans were labeled.

I understood spacy better after reading this tutorial: https://cees-roele.medium.com/detecting-toxic-spans-with-spacy-c5533786bbf8
