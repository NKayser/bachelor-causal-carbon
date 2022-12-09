import re

from MyCode.scripts.consts import INPUT_PATH, TEXTCAT_MODEL_PATH, SPANCAT_MODEL_PATH, PRETRAINED_NER_MODEL, \
    TECHNOLOGY_CATEGORIES
from MyCode.scripts.spacy_utility_functions import apply_textcat, apply_spancat, apply_pretrained_ner, apply_sentencizer
from MyCode.scripts.utils import get_positive_article_ids, get_all_entities_by_label, \
    get_more_precise_locations, read_input_file, get_sent_of_ent, get_span_labels_of_sentence, \
    get_all_entities_in_sentence


class Article:
    text = None
    metadata = None

    textcat_prediction = None
    sents = None
    spans = None
    ents = None
    ents_by_type = None

    def __init__(self, metadata, text, texcat_prediction=None, sents=None, spans=None, ents=None):
        self.metadata = metadata
        self.text = text
        self.textcat_prediction = texcat_prediction
        self.sents = sents
        self.spans = spans
        self.ents = ents
        self.ents_by_type = get_all_entities_by_label(self.ents)

    @classmethod
    def from_dict(cls, precomputed_dict, include_predictions=True):
        if include_predictions:
            article_text = precomputed_dict["text"]
            sents = Charspan.from_dict_array(precomputed_dict["sents"], article_text, "sent")
            spans = Charspan.from_dict_array(precomputed_dict["predicted_sent_spans"], article_text, "span")
            ents = Charspan.from_dict_array(precomputed_dict["entities"], article_text, "ent")
            return cls(precomputed_dict, article_text, precomputed_dict["textcat_prediction"], sents, spans, ents)
        else:
            return cls(precomputed_dict, precomputed_dict["text"])

    @classmethod
    def from_article(cls, article_id, include_predictions=True, file_path=INPUT_PATH):
        json_list = read_input_file(file_path)

        for article_data in json_list:
            if article_data["id"] == article_id:
                return cls.from_dict(article_data, include_predictions)

        print("Article with id " + str(article_id) + " could not be found.")
        assert False

    def preprocess_spacy(self, textcat_model=TEXTCAT_MODEL_PATH, spancat_model=SPANCAT_MODEL_PATH,
                         ner_model=PRETRAINED_NER_MODEL):
        self.textcat_prediction = apply_textcat(self.text, textcat_model)
        self.sents = Charspan.from_dict_array(apply_sentencizer(self.text, spancat_model), self.text, "sent")
        self.spans = Charspan.from_dict_array(apply_spancat(self.text, spancat_model), self.text, "span")
        self.ents = Charspan.from_dict_array(apply_pretrained_ner(self.text, ner_model), self.text, "ent")
        self.ents_by_type = get_all_entities_by_label(self.ents)

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
        return sorted_counts

    def get_locations(self):
        #all_spans = get_all_entities_by_type(self.text, self.spans)
        #if "sent_location" in all_spans.keys():
        #    print(all_spans["sent_location"])
        better_locations = get_more_precise_locations(self.ents_by_type["GPE"])
        return better_locations

    def get_financial_information(self):
        # self.ents_by_type["MONEY"]
        money_ents = list(filter(lambda ent: ent.label == "MONEY", self.ents))
        money_spans = list(filter(lambda span: span.label == "sent_financial information", self.spans))

        # fix problem where "€50m" -> "50" and "more than €160m" -> "more than €160"

        for ent in money_ents:
            sent = get_sent_of_ent(ent, self.sents)
            print("span labels:")
            print(get_span_labels_of_sentence(self.spans, sent))
            print("entities:")
            print([str(ent) for ent in get_all_entities_in_sentence(self.ents, sent)])
            print("sent text:")
            print(sent.text)
            print("ent text:")
            print(ent.text)


        print("all money ents:")
        print([str(ent) for ent in money_ents])
        print("all money spans:")
        print([str(ent) for ent in money_spans])

        return(money_ents)


class Charspan:
    id = None
    start_offset = None
    end_offset = None
    type = None
    label = None
    text = None
    article_text = None

    def __init__(self, span_id, start_offset, end_offset, label, article_text, span_type):
        if start_offset < 0 or end_offset > len(article_text):
            print("Entity boundaries outside of text")
            assert False

        self.id = span_id
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.type = span_type
        self.label = label
        self.article_text = article_text
        self.text = self.get_text()

    @classmethod
    def from_dict(cls, input_dict, article_text, span_type):
        if "label" in input_dict.keys():
            label = input_dict["label"]
        else:
            label = None
        return cls(input_dict["id"], input_dict["start_offset"], input_dict["end_offset"], label, article_text,
                   span_type)

    @classmethod
    def from_dict_array(cls, arr, article_text, span_type):
        return [cls.from_dict(span_dict, article_text, span_type) for span_dict in arr]

    def get_text(self, padding=0):
        start_char = self.start_offset - padding
        end_char = self.end_offset + padding
        if start_char < 0:
            start_char = 0
        if end_char > len(self.article_text):
            end_char = len(self.article_text)
        return self.article_text[start_char:end_char]

    def __str__(self):
        return str({"label": self.label, "text": self.text})


if __name__ == '__main__':
    positive_ids = get_positive_article_ids()
    article = Article.from_article(positive_ids[1]) # e.g. 6389
    print(article.text)
    #print(article.get_technology_cats())
    #print(article.get_locations())  # some weird locations for number 25
                                    # Czech Republic points to specific location in country for some reason
                                    # error in 80
    print(article.get_financial_information())
    #print(article.textcat_prediction)
    #print(article.spans)
    #print(article.ents)
