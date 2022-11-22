# bachelor-causal-carbon

## To run spacy-causal-carbon project:

* cd to MyCode/spacy-causal-carbon
* `pip install -r requirements.txt`
* A cuda version and torch is also needed, but these need to be compatible with each other. Get installation command from https://pytorch.org/get-started/locally/
* Maybe required in the future, maybe also en_core_web_lg, en_core_web_trf: `python -m spacy download en_core_web_sm`
* `python -m spacy project assets`
* `python -m spacy project run corpus`
* When changing base_config: `python -m spacy init fill-config configs/base_config.cfg configs/config.cfg`
* `python -m spacy project run train`
* `python -m spacy project run evaluate

There is also a display_spans script to get a better idea of how the spans were labeled.

I understood spacy better after reading this tutorial: https://cees-roele.medium.com/detecting-toxic-spans-with-spacy-c5533786bbf8


## To run spacy_textcat project:

* Run make_corpus script
* `python -m spacy train configs/config.cfg --gpu-id 0 -o models`
* `python -m spacy evaluate models/model-best corpus/textcat_test.spacy --gpu-id 0`