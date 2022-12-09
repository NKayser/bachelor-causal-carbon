import re
import spacy

from MyCode.scripts.consts import INPUT_PATH, TEXTCAT_MODEL_PATH, SPANCAT_MODEL_PATH, PRETRAINED_NER_MODEL, \
    TECHNOLOGY_CATEGORIES
from MyCode.scripts.spacy_utility_functions import apply_textcat, apply_spancat, apply_pretrained_ner, apply_sentencizer
from MyCode.scripts.utils import get_positive_article_ids, get_all_entities_by_label, \
    get_more_precise_locations, read_input_file, get_span_labels_of_sentence, \
    get_all_entities_in_sentence, ent_is_in_sent


class Article:
    text = None
    metadata = None

    textcat_prediction = None
    sents = None
    spans = None
    ents = None
    ents_by_type = None

    def __init__(self, metadata, text, texcat_prediction=None, sents=None, spans=None, ents=None):
        assert text is not None
        self.metadata = metadata
        self.text = text
        self.textcat_prediction = texcat_prediction
        self.sents = sents
        self.spans = spans
        self.ents = ents
        self.ents_by_type = get_all_entities_by_label(self.ents)

    @classmethod
    def from_dict(cls, precomputed_dict, include_predictions=True):
        article_text = precomputed_dict["text"]
        if include_predictions:
            sents = Charspan.from_dict_array(precomputed_dict["sents"], article_text, "sent")
            spans = Charspan.from_dict_array(precomputed_dict["predicted_sent_spans"], article_text, "span")
            ents = Charspan.from_dict_array(precomputed_dict["entities"], article_text, "ent")
            return cls(precomputed_dict, article_text, precomputed_dict["textcat_prediction"], sents, spans, ents)
        else:
            return cls(precomputed_dict, article_text)

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

    def get_sent_of_ent(self, ent):
        for i in range(0, len(self.sents)):
            sent = self.sents[i]
            if ent_is_in_sent(ent, sent):
                if ent.start_offset < sent.start_offset:
                    self.sents[i-1].set_new_offset(self.sents[i-1].start_offset, self.sents[i].end_offset)
                if ent.end_offset > sent.end_offset:
                    self.sents[i].set_new_offset(self.sents[i].start_offset, self.sents[i+1].end_offset)
                return sent
        print("Error with loading document: entity " + ent.text + " was not found in sentences.")
        assert False

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
        ner_length = len(money_ents)
        print([ent.text for ent in money_ents])
        money_spans = list(filter(lambda span: span.label == "sent_financial information", self.spans))

        keywords = [("invest", 3), ("project", 2), ("technology", 2), ("plant", 2), ("environment", 1), ("sustain", 1)] # with weights
        #additional_patterns = ["(EUR|eur|euro|euros|Euro|€|\u20ac|USD|Usd|usd|\$|US\$|CAN|CAN\$|CAD|cad|can|CHF|PLN|\u00a3) ?(\d+([\.,]?\d*)*)[-–]?(\d*([\.,]?\d*)*)\+? ?(million|mio|mln|m|billion|bn|b|thousand)\.?",
        #                       "(\d+([\.,]?\d*)*)[-–]?(\d*([\.,]?\d*)*)\+? ?(m|mio|mln|million|b|bn|billion|thousand| )\.? ?(EUR|eur|euro|euros|Euro|€|\u20ac|USD|Usd|usd|\$|US\$|CAN|CAN\$|CAD|cad|can|CHF|PLN|\u00a3)"]
        additional_patterns = ["(EUR|Eur|eur|euro|euros|Euro|€|\u20ac|USD|Usd|usd|\$|US\$|us\$|Us\$|CAN|CAN\$|CAD|CAD\$|cad|cad\$|can|can\$|Cad|Cad\$|Can|Can\$|CHF|Chf|chf|PLN|pln|Pln|\u00a3) ?(\d+([\.,]?\d*)*)[-–]?(\d*([\.,]?\d*)*)\+? ?(million|mio|mln|m|billion|bn|b|thousand)",
                                       "(\d+([\.,]?\d*)*)[-–]?(\d*([\.,]?\d*)*)\+? ?(m|mio|mln|million|b|bn|billion|thousand| )\.? ?(EUR|Eur|eur|euro|euros|Euro|€|\u20ac|USD|Usd|usd|\$|US\$|us\$|Us\$|CAN|CAN\$|CAD|CAD\$|cad|cad\$|can|can\$|Cad|Cad\$|Can|Can\$|CHF|Chf|chf|PLN|pln|Pln|\u00a3)"]

        # fix problem where "€50m" -> "50" and "more than €160m" -> "more than €160"
        for pattern in additional_patterns:
            matches = re.finditer(pattern, self.text)
            count_matches = 0
            for match in matches:
                print(self.text[match.start():match.end()])
                count_matches += 1
                match_text = self.text[match.start():match.end()]
                if match_text in [ent.text for ent in money_ents]:
                    continue
                match_in_existing = False
                for i in range(0, len(money_ents)):
                    me_start = money_ents[i].start_offset
                    me_end = money_ents[i].end_offset
                    assert me_start < me_end and match.start() < match.end()
                    if not (me_end <= match.start() or match.end() <= me_start):
                        money_ents[i].set_new_offset(min(me_start, match.start()), max(me_end, match.end()))
                        match_in_existing = True
                if not match_in_existing:
                    new_money_ent = Charspan(match.start(), match.end(), "MONEY", self.text, "ent")
                    money_ents.append(new_money_ent)

            print("Len of NER matches before: " + str(ner_length) + ", len of re matches: " + str(count_matches)
                  + ", len of NER matches after: " + str(len(money_ents)))
            #money_ents.append(Charspan.from_dict_array())

        for ent in money_ents:
            print(str(ent))
            sent = self.get_sent_of_ent(ent)
            ents_of_sent = get_all_entities_by_label(get_all_entities_in_sentence(self.ents, sent))
            span_labels = get_span_labels_of_sentence(self.spans, sent)
            keyword_number = 0
            for keyword, weight in keywords:
                count = len(re.findall(keyword, sent.text))
                keyword_number += count * weight

            #print(sent.text)
            #print(keyword_number)
            #print(span_labels)
            #print([str(ent) for ent in ents_of_sent])
            #print(ent.text)


        #print("all money ents:")
        #print([str(ent) for ent in money_ents])
        #print("all money spans:")
        #print([str(ent) for ent in money_spans])

        return(money_ents)


class Charspan:
    id = None
    start_offset = None
    end_offset = None
    type = None
    label = None
    text = None
    article_text = None

    def __init__(self, start_offset, end_offset, label, article_text, span_type, span_id=None):
        assert article_text is not None
        self.article_text = article_text
        self.set_new_offset(start_offset, end_offset)
        self.id = span_id
        self.type = span_type
        self.label = label

    @classmethod
    def from_dict(cls, input_dict, article_text, span_type):
        if "label" in input_dict.keys():
            label = input_dict["label"]
        else:
            label = None
        return cls(input_dict["start_offset"], input_dict["end_offset"], label, article_text,
                   span_type, input_dict["id"])

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

    def set_new_offset(self, new_start_offset, new_end_offset):
        if new_start_offset < 0 or new_end_offset > len(self.article_text):
            print("Entity boundaries outside of text")
            assert False
        self.start_offset = new_start_offset
        self.end_offset = new_end_offset
        self.text = self.get_text()


if __name__ == '__main__':
    positive_ids = get_positive_article_ids()
    article = Article.from_article(positive_ids[4]) # e.g. 6389
    #print(article.text)
    #print(article.get_technology_cats())
    #print(article.get_locations())  # some weird locations for number 25
                                    # Czech Republic points to specific location in country for some reason
                                    # error in 80
    print(article.get_financial_information())
    #print(article.textcat_prediction)
    #print(article.spans)
    #print(article.ents)
