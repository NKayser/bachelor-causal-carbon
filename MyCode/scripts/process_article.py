import json

from MyCode.scripts.spacy_utility_functions import apply_textcat, apply_spancat, apply_pretrained_ner

INPUT_PATH = "../data/labels_and_predictions.jsonl"
TEXTCAT_MODEL_PATH = "../textcat/models/cval_2/model-best"
SPANCAT_MODEL_PATH = "../spancat/models/model-best"
PRETRAINED_NER_MODEL = "en_core_web_trf"
TEXTCAT_THRESHOLD = 0.5


class Article:
    text = None
    metadata = None

    textcat_prediction = None
    spans = None
    ents = None

    def fill_from_dict(self, precomputed_dict, include_predictions=True):
        self.metadata = precomputed_dict
        self.text = precomputed_dict["text"]

        if include_predictions:
            self.textcat_prediction = precomputed_dict["textcat_prediction"]
            self.spans = precomputed_dict["predicted_sent_spans"]
            self.ents = precomputed_dict["entities"]

    def fill_from_article(self, article_id, include_predictions=True, file_path=INPUT_PATH):
        with open(file_path, "r", encoding="utf-8") as input_file:
            json_list = list(input_file)

        for json_str in json_list:
            article_data = json.loads(json_str)
            if article_data["id"] == article_id:
                self.fill_from_dict(article_data, include_predictions)
                return self

        print("Article with id " + str(article_id) + " could not be found.")
        assert False

    def preprocess_spacy(self, textcat_model=TEXTCAT_MODEL_PATH, spancat_model=SPANCAT_MODEL_PATH,
                         ner_model=PRETRAINED_NER_MODEL):
        self.textcat_prediction = apply_textcat(self.text, textcat_model)
        self.spans = apply_spancat(self.text, spancat_model)
        self.ents = apply_pretrained_ner(self.text, ner_model)


if __name__ == '__main__':
    article = Article()
    article.fill_from_article(6389)
    print(article.textcat_prediction)
    print(article.spans)
    print(article.ents)
