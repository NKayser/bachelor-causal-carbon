import json
import re

from MyCode.scripts.consts import INPUT_PATH, TEXTCAT_MODEL_PATH, SPANCAT_MODEL_PATH, PRETRAINED_NER_MODEL, \
    TECHNOLOGY_CATEGORIES
from MyCode.scripts.spacy_utility_functions import apply_textcat, apply_spancat, apply_pretrained_ner
from MyCode.scripts.utils import get_positive_article_ids


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

    def get_technology_cats(self, categories=TECHNOLOGY_CATEGORIES):
        # define the categories and keywords
        # later: could define weights of keywords
        # could use tree structure
        # could use logical rules ("word1 AND word2 OR word3 AND NOT word4")

        # initialize the counts for each category
        counts = {c: 0 for c in categories}

        # search for each keyword in the text
        for c, keywords in categories.items():
            for keyword in keywords:
                count = len(re.findall(keyword, self.text))
                counts[c] += count

        # print the counts for each category
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        sorted_counts = [x for x in sorted_counts if x[1] != 0]
        print(sorted_counts)


if __name__ == '__main__':
    positive_ids = get_positive_article_ids()
    article = Article()
    article.fill_from_article(positive_ids[100]) # e.g. 6389
    print(article.text)
    article.get_technology_cats()
    #print(article.textcat_prediction)
    #print(article.spans)
    #print(article.ents)
