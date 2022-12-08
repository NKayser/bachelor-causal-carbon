# bachelor-causal-carbon

## To run spacy-causal-carbon project:

* `cd .\MyCode\spacy-causal-carbon\`
* `pip install -r requirements.txt`
* A cuda version and torch is also needed, but these need to be compatible with each other. Get installation command from https://pytorch.org/get-started/locally/
* Maybe required in the future, maybe also en_core_web_lg, en_core_web_trf: `python -m spacy download en_core_web_sm`
* `python -m spacy project assets`
* `python -m spacy project run corpus`
* When changing base_config: `python -m spacy init fill-config configs/base_config.cfg configs/config.cfg`
* `python -m spacy project run train`
* `python -m spacy project run evaluate`

There is also a display_spans script to get a better idea of how the spans were labeled.

I understood spacy better after reading this tutorial: https://cees-roele.medium.com/detecting-toxic-spans-with-spacy-c5533786bbf8


## To run spacy_textcat project:

* `cd .\MyCode\spacy_textcat\`
* Run make_corpus script

Train:
* `python -m spacy train configs/config.cfg --gpu-id 0 -o models`

Evaluate:
* `python -m spacy evaluate models/model-best corpus/textcat_test.spacy --gpu-id 0`

Crossvalidation:
* A workflow that will fill the config file, create training data files, train and evaluate all buckets sequentially
* `python -m spacy project run crossvalidation`


## labels_and_predictions input file explained:

* Attributes from database, self-explanatory:
  * id, text, company_name, published_on, title, emissions_rank, language, article_category, preview_text, article_url, pdf_url, error_text, carbon_in_title, carbon_in_text
* entities: as predicted by pretrained NER algorithm of en_core_web_trf
  * id, label, start_offset, end_offset
* relations: labeled by me (in 118 articles)
  * id, from_id, to_id, type
* label: labeled by me. "positive", "negative" or "unsure" if article about investment in carbon abatement
* textcat_prediction: label predicted by best textcat model (cval_2), to save computing time
* labeled_entities: labeled by me. includes "core reference", "location", "effect", "cause", "status", "technology", "emissions", "timeline", "financial information", "carbon investment"
* predicted_sent_spans: predicted sentence labels by spancat model, to save computing time. Trained on labeled_entities with boundaries expanded to sentences.
  * same labels as labeled_entities but with "sent_" prefix
* dataset: train, test, val or null. Which dataset article belonged to for training of model used for "textcat_prediction" value