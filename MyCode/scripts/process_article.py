import re

from MyCode.scripts.Charspan import Charspan
from MyCode.scripts.consts import INPUT_PATH, TEXTCAT_MODEL_PATH, SPANCAT_MODEL_PATH, PRETRAINED_NER_MODEL, \
    TECHNOLOGY_CATEGORIES
from MyCode.scripts.process_for_property_utils import get_additional_money_ents
from MyCode.scripts.spacy_utility_functions import apply_textcat, apply_spancat, apply_pretrained_ner, apply_sentencizer
from MyCode.scripts.utils import get_positive_article_ids, get_all_entities_by_label, \
    get_more_precise_locations, read_input_file, ent_is_in_sent, filter_ents, get_ents_of_sent, \
    get_span_labels_of_sentence


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

    def get_investment_information(self):
        finance_ents = self.get_financial_information()     # "MONEY"
        technology_ents = self.get_technology_ents()        # "technology"
        location_ents = self.get_location_ents()            # "GPE"
        emissions_ents = self.get_emissions_ents()          # "QUANTITY"
        time_ents = self.get_time_ents()                    # "TIME"
        parsed_ents = [finance_ents, technology_ents, location_ents, emissions_ents, time_ents]

        for sent in self.sents:
            print("\nSentence:")
            print(sent.text)
            sent_ents = [get_ents_of_sent(ents, sent) for ents in parsed_ents]
            print([[ent.text for ent in ents] for ents in sent_ents])


    def get_technology_ents(self, categories=TECHNOLOGY_CATEGORIES):
        technology_ents = []
        for c, keywords in categories.items():
            for keyword in keywords:
                for match in re.finditer(keyword, self.text):
                    new_tech_ent = Charspan(match.start(), match.end(), "technology", self.text, "ent")
                    technology_ents.append(new_tech_ent)
        return technology_ents

    def get_weighted_technology_cats(self, categories=TECHNOLOGY_CATEGORIES):
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

    def get_location_ents(self):
        return filter_ents(self.ents, "GPE")

    def get_weighted_locations(self):
        better_locations = get_more_precise_locations(self.ents_by_type["GPE"])
        return better_locations

    def get_financial_information(self):
        money_ents = filter_ents(self.ents, "MONEY")
        money_ents = get_additional_money_ents(self.text, money_ents)
        return money_ents

    def get_emissions_ents(self):
        # Problem: "kW/h" only recognized as "kW"
        emissions_ents = filter_ents(self.ents, "QUANTITY")
        return emissions_ents

    def get_time_ents(self):
        return filter_ents(self.ents, "DATE")


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
