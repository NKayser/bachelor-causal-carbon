import re

import spacy

from MyCode.scripts.consts import INPUT_PATH, TEXTCAT_MODEL_PATH, SPANCAT_MODEL_PATH, PRETRAINED_NER_MODEL, \
    TECHNOLOGY_CATEGORIES, WEIGHTED_SENT_KEYWORDS
from MyCode.scripts.process_for_property_utils import get_additional_money_ents, get_weighted_technology_cats
from MyCode.scripts.spacy_utility_functions import apply_textcat, apply_spancat, apply_pretrained_ner, apply_sentencizer
from MyCode.scripts.utils import get_positive_article_ids, get_all_entities_by_label, \
    get_more_precise_locations, read_input_file, ent_is_in_sent, filter_ents, get_ents_of_sent


class Article:
    nlp = spacy.blank("en")
    doc = None
    metadata = None

    textcat_prediction = None

    def __init__(self, text, metadata=None, texcat_prediction=None, sents=None, spans=None, ents=None):
        assert text is not None
        self.doc = self.nlp(text)
        self.metadata = metadata
        self.textcat_prediction = texcat_prediction
        self.set_sents(sents)
        self.doc.spans["sc"] = self.dict_to_charspan_array(spans)
        self.doc.ents = self.dict_to_charspan_array(ents)

    @classmethod
    def from_dict(cls, precomputed_dict, include_predictions=True):
        article_text = precomputed_dict["text"]
        if include_predictions:
            return cls(article_text, precomputed_dict, precomputed_dict["textcat_prediction"],
                       precomputed_dict["sents"], precomputed_dict["predicted_sent_spans"],
                       precomputed_dict["entities"])
        else:
            return cls(article_text, precomputed_dict)

    @classmethod
    def from_article(cls, article_id, include_predictions=True, file_path=INPUT_PATH):
        json_list = read_input_file(file_path)

        for article_data in json_list:
            if article_data["id"] == article_id:
                return cls.from_dict(article_data, include_predictions)

        print("Article with id " + str(article_id) + " could not be found.")
        assert False

    def set_sents(self, in_dict):
        char_span_sents = self.dict_to_charspan_array(in_dict)
        for char_span_sent in char_span_sents:
            char_span_sent[0].is_sent_start = True

    def dict_to_charspan_array(self, in_dict):
        spans = []
        for entry in in_dict:
            if "label" in entry.keys():
                spans.append(self.doc.char_span(entry["start_offset"], entry["end_offset"], label=entry["label"],
                                                alignment_mode="expand"))
            else:
                spans.append(self.doc.char_span(entry["start_offset"], entry["end_offset"],
                                                alignment_mode="expand"))
        return spans

    def preprocess_spacy(self, textcat_model=TEXTCAT_MODEL_PATH, spancat_model=SPANCAT_MODEL_PATH,
                         ner_model=PRETRAINED_NER_MODEL):
        self.textcat_prediction = apply_textcat(self.doc.text, textcat_model)
        self.doc.sents = self.dict_to_charspan_array(apply_sentencizer(self.doc.text, spancat_model))
        self.doc.spans["sc"] = self.dict_to_charspan_array(apply_spancat(self.doc.text, spancat_model))
        self.doc.ents = self.dict_to_charspan_array(apply_pretrained_ner(self.doc.text, ner_model))

    def get_sent_of_ent(self, ent):
        for i in range(0, len(self.doc.sents)):
            sent = self.doc.sents[i]
            if ent_is_in_sent(ent, sent):
                if ent.start_offset < sent.start_offset:
                    self.doc.sents[i-1].set_new_offset(self.doc.sents[i-1].start_offset, self.doc.sents[i].end_offset)
                if ent.end_offset > sent.end_offset:
                    self.doc.sents[i].set_new_offset(self.doc.sents[i].start_offset, self.doc.sents[i+1].end_offset)
                return sent
        print("Error with loading document: entity " + ent.text + " was not found in sentences.")
        assert False

    def get_investment_information(self):
        finance_ents = self.get_financial_information()     # "MONEY"
        technology_ents = self.get_technology_ents()        # "TECHWORD"
        location_ents = self.get_location_ents()            # "GPE"
        emissions_ents = self.get_emissions_ents()          # "QUANTITY", "PERCENT"
        time_ents = self.get_date_ents()                    # "DATE"
        parsed_ents = [finance_ents, technology_ents, location_ents, emissions_ents, time_ents]
        weighted_sents = self.get_weighted_sents()

        for sent, weight in weighted_sents:
            sent_ents = [get_ents_of_sent(ents, sent) for ents in parsed_ents]
            count_ents = [len(arr) for arr in sent_ents]
            sum_ents = sum(count_ents)
            technology_cats_of_sent = get_weighted_technology_cats(sent.text)
            if sum_ents == 0:
                continue
            print("\n")
            print(sent.text)
            print([[ent.text for ent in ents] for ents in sent_ents])
            print("tech cats: ", technology_cats_of_sent)
            print("sentence weight: ", weight)
            print("number of ents:  ", sum_ents)

            # weighted_sents = [(sent, weight / total) for sent, weight in weighted_sents]

    def get_weighted_sents(self, weighted_keywords=WEIGHTED_SENT_KEYWORDS):
        weighted_sents = []
        total = 0
        for sent in self.doc.sents:
            sent_weight = 0
            for keyword, weight in weighted_keywords:
                sent_weight += len(re.findall(keyword, sent.text)) * weight
            weighted_sents.append((sent, sent_weight))
            total += sent_weight
        return weighted_sents

    def get_technology_ents(self, categories=TECHNOLOGY_CATEGORIES):
        technology_ents = []
        for c, keywords in categories.items():
            for keyword in keywords:
                for match in re.finditer(keyword, self.doc.text):
                    new_tech_ent = self.doc.char_span(match.start(), match.end(), "TECHWORD", alignment_mode="expand")
                    technology_ents.append(new_tech_ent)
        return technology_ents

    def get_weighted_technology_cats(self):
        get_weighted_technology_cats(self.doc.text)

    def get_location_ents(self):
        return filter_ents(self.doc.ents, "GPE")

    def get_weighted_locations(self):
        better_locations = get_more_precise_locations(get_all_entities_by_label(self.doc.ents)["GPE"])
        return better_locations

    def get_financial_information(self):
        money_ents = filter_ents(self.doc.ents, "MONEY")
        money_ents = get_additional_money_ents(self.doc, money_ents)
        return money_ents

    def get_emissions_ents(self):
        # Problem: "kW/h" only recognized as "kW"
        emissions_ents = filter_ents(self.doc.ents, "QUANTITY")
        emissions_ents = emissions_ents + filter_ents(self.doc.ents, "PERCENT")
        return emissions_ents

    def get_date_ents(self):
        return filter_ents(self.doc.ents, "DATE")


if __name__ == '__main__':
    positive_ids = get_positive_article_ids()
    id = positive_ids[-1]
    print("id " + str(id))
    article = Article.from_article(id) # e.g. 6389
    article.get_investment_information()
    #article.get_financial_information()
    #print(article.text)
    #print(article.get_technology_cats())
    #print(article.get_locations())  # some weird locations for number 25
                                    # Czech Republic points to specific location in country for some reason
                                    # error in 80
    #print(article.textcat_prediction)
    #print(article.spans)
    #print(article.ents)
