import json
import re

import spacy
from spacy import registry
from spacy.tokens import SpanGroup
from tqdm import tqdm

from MyCode.entity_categorization.scripts.custom_span_suggester import build_custom_suggester
from MyCode.scripts.consts import INPUT_PATH, TEXTCAT_MODEL_PATH, SPANCAT_MODEL_PATH, PRETRAINED_NER_MODEL, \
    TECHNOLOGY_CATEGORIES, WEIGHTED_SENT_KEYWORDS, ENT_MODEL_PATH
from MyCode.scripts.process_for_property_utils import get_additional_money_ents, get_weighted_technology_cats
from MyCode.scripts.spacy_utility_functions import apply_textcat, apply_spancat, apply_pretrained_ner, \
    apply_sentencizer
from MyCode.scripts.utils import get_positive_article_ids, get_all_entities_by_label, \
    parse_location, read_input_file, ent_is_in_sent, filter_ents, get_ents_of_sent, opposite_filter_ents, \
    sort_by_ent_cat, parse_money, parse_percent, parse_quantity, parse_time, get_more_precise_locations, \
    parse_weighted_tech, simple_loc_parse, filter_none


@registry.misc("article_all_ent_suggester.v1")
def suggester():
    return build_custom_suggester(balance=False, input_path=INPUT_PATH)


class Article:
    nlp = spacy.blank("en")
    doc = None
    metadata = None

    textcat_prediction = None
    ent_cats = None

    ent_cat_doc = None

    def __init__(self, text, metadata=None, texcat_prediction=None, sents=None, spans=None, ents=None):
        assert text is not None
        self.doc = self.nlp(text)
        self.metadata = metadata
        self.textcat_prediction = texcat_prediction
        if sents is not None:
            self.set_sents(sents)
        if spans is not None:
            self.doc.spans["sc"] = self.dict_to_charspan_array(spans)
        else:
            self.doc.spans["sc"] = []
        if ents is not None:
            self.doc.spans["sc"] += self.dict_to_charspan_array(ents)

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

    def apply_ner(self, ner_model=PRETRAINED_NER_MODEL):
        self.doc.spans["sc"] += self.dict_to_charspan_array(apply_pretrained_ner(self.doc.text, ner_model))

    def preprocess(self, textcat_model=TEXTCAT_MODEL_PATH, spancat_model=SPANCAT_MODEL_PATH,
                   ner_model=PRETRAINED_NER_MODEL, ent_cat_model=ENT_MODEL_PATH):
        self.textcat_prediction = apply_textcat(self.doc.text, textcat_model)
        self.set_sents(apply_sentencizer(self.doc.text, spancat_model))
        self.doc.spans["sc"] = self.dict_to_charspan_array(apply_spancat(self.doc.text, spancat_model))
        self.doc.spans["sc"] += self.dict_to_charspan_array(apply_pretrained_ner(self.doc.text, ner_model))
        self.set_money_ents()
        self.load_ent_cat_model(ent_cat_model)


    def get_investment_information_v1(self, threshold=-1.0, parse=True, parse_loc=True):
        # need to preprocess_spacy before
        finance_ents = self.set_money_ents()                # "MONEY"
        weighted_tech_ents = self.get_weighted_technology_cats()
        location_ents = self.get_location_ents()            # "GPE"
        quantity_ents = filter_ents(self.doc.spans["sc"], "QUANTITY")
        percent_ents = filter_ents(self.doc.spans["sc"], "PERCENT")
        fac_ents = filter_ents(self.doc.spans["sc"], "FAC")
        time_ents = self.get_date_ents()                    # "DATE"
        ent_cats = self.get_ent_cats()                      # "positive" or "negative", 2d array with confidence scores

        # if faculty name is also in locations, add it again to locations (might increase confidence score)
        for ent in fac_ents:
            if ent.text in [loc_ent.text for loc_ent in location_ents]:
                location_ents.append(ent)

        for ent in finance_ents:
            if "boe" in ent.text.lower() or "nm3" in ent.text.lower():
                finance_ents.remove(ent)
                quantity_ents.append(ent)

        if parse:
            if parse_loc:
                loc_parse_func = parse_location
            else:
                loc_parse_func = simple_loc_parse
            info = {"technology": filter_none(list(map(parse_weighted_tech, weighted_tech_ents))),
                "money": filter_none(list(map(parse_money, sort_by_ent_cat(finance_ents, ent_cats, threshold)))),
                "location": filter_none(list(map(loc_parse_func, sort_by_ent_cat(location_ents, ent_cats, threshold)))),
                "emissions_percent": filter_none(list(map(parse_percent, sort_by_ent_cat(percent_ents, ent_cats, threshold)))),
                "emissions_quantity": filter_none(list(map(parse_quantity, sort_by_ent_cat(quantity_ents, ent_cats, threshold)))),
                "time": filter_none(list(map(parse_time, sort_by_ent_cat(time_ents, ent_cats, threshold))))}
        else:
            info = {"technology": filter_none(weighted_tech_ents),
                    "money": filter_none(list(sort_by_ent_cat(finance_ents, ent_cats, threshold))),
                    "location": filter_none(list(sort_by_ent_cat(location_ents, ent_cats, threshold))),
                    "emissions_percent": filter_none(list(sort_by_ent_cat(percent_ents, ent_cats, threshold))),
                    "emissions_quantity": filter_none(list(sort_by_ent_cat(quantity_ents, ent_cats, threshold))),
                    "time": filter_none(list(sort_by_ent_cat(time_ents, ent_cats, threshold)))}

        out_obj = {"metadata": self.metadata,
                   "textcat_prediction": self.textcat_prediction,
                   "parsed_info": info}
        return out_obj

    def get_investment_information_v2(self):
        finance_ents = self.set_money_ents()                # "MONEY"
        technology_ents = self.set_technology_ents()        # "TECHWORD"
        location_ents = self.get_location_ents()            # "GPE"
        emissions_ents = self.get_emissions_ents()          # "QUANTITY", "PERCENT"
        time_ents = self.get_date_ents()                    # "DATE"
        ent_cats = self.get_ent_cats()                      # "positive" or "negative"
        parsed_ents = [finance_ents, technology_ents, location_ents, emissions_ents, time_ents, ent_cats]
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

    def load_ent_cat_model(self, ent_cat_model):
        config_path = ent_cat_model + "/config.cfg"
        try:
            with open(config_path, "r") as config:
                lines = config.readlines()
            with open(config_path, "w") as config:
                for line in lines:
                    if line.strip("\n") not in ['balance = "True"', 'input_path = "assets/labels_and_predictions.jsonl"']:
                        config.write(line)

            spacy.prefer_gpu(0)
            nlp2 = spacy.load(ent_cat_model)
            self.ent_cat_doc = nlp2(self.doc.text)
        except ValueError:
            print("Loading entity categorization failed")
            self.ent_cat_doc = None

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

    def set_technology_ents(self, categories=TECHNOLOGY_CATEGORIES):
        technology_ents = SpanGroup(self.doc)
        for c, keywords in categories.items():
            for keyword in keywords:
                for match in re.finditer(keyword, self.doc.text):
                    new_tech_ent = self.doc.char_span(match.start(), match.end(), "TECHWORD", alignment_mode="expand")
                    technology_ents.append(new_tech_ent)
        self.doc.spans["sc"] = self.doc.spans["sc"] + technology_ents
        return technology_ents

    def get_weighted_technology_cats(self):
        return get_weighted_technology_cats(self.doc.text)

    def get_location_ents(self):
        return filter_ents(self.doc.spans["sc"], "GPE")

    def get_weighted_locations(self):
        better_locations = get_more_precise_locations(get_all_entities_by_label(self.doc.spans["sc"])["GPE"])
        return better_locations

    def set_money_ents(self):
        money_ents = filter_ents(self.doc.spans["sc"], "MONEY")
        money_ents = get_additional_money_ents(self.doc, money_ents)
        self.doc.spans["sc"] = opposite_filter_ents(self.doc.spans["sc"], "MONEY") + money_ents
        return money_ents

    def get_emissions_ents(self):
        # Problem: "kW/h" only recognized as "kW"
        emissions_ents = filter_ents(self.doc.spans["sc"], "QUANTITY")
        emissions_ents = emissions_ents + filter_ents(self.doc.spans["sc"], "PERCENT")
        return emissions_ents

    def get_date_ents(self):
        return filter_ents(self.doc.spans["sc"], "DATE")

    def get_ent_cats(self):
        if self.ent_cat_doc is None:
            return SpanGroup(self.doc)
        spans = self.ent_cat_doc.spans["sc"]
        return spans


def parse_and_save_all_articles(out_path="outputs/parsed_data3.jsonl", start_at_id=6001):
    articles = read_input_file()
    with open(out_path, "a", encoding="utf-8") as out_file:
        for article_obj in tqdm(articles):
            if article_obj["id"] < start_at_id:
                continue
            article = Article.from_article(article_obj["id"])
            article.preprocess()
            article.set_money_ents()
            out = json.dumps(article.get_investment_information_v1(), ensure_ascii=False)
            out_file.write(out + "\n")


def redo_money_and_quantity_parse(in_path="outputs/parsed_data.jsonl", out_path="outputs/parsed_data2.jsonl", start_at_id=6001):
    articles = read_input_file(in_path)
    parsed_ids = []
    with open(out_path, "a", encoding="utf-8") as out_file:
        for article_obj in tqdm(articles):
            article_id = article_obj["metadata"]["id"]
            if article_id < start_at_id or article_id in parsed_ids:
                print("skipped id", article_id)
                continue
            #print("id", article_id)
            money_arr = [(x["original"], x["confidence"]) for x in article_obj["parsed_info"]["money"]]
            quant_arr = [(x["original"], x["confidence"]) for x in article_obj["parsed_info"]["emissions_quantity"]]
            time_arr = [(x["original"], x["confidence"]) for x in article_obj["parsed_info"]["time"]]
            money = filter_none(list(map(parse_money, money_arr)))
            quant = filter_none(list(map(parse_quantity, quant_arr)))
            time = filter_none(list(map(parse_time, time_arr)))
            article_obj["parsed_info"]["money"] = money
            article_obj["parsed_info"]["emissions_quantity"] = quant
            article_obj["parsed_info"]["time"] = time
            out = json.dumps(article_obj, ensure_ascii=False)
            out_file.write(out + "\n")
            parsed_ids.append(article_id)


if __name__ == '__main__':
    positive_ids = get_positive_article_ids()
    article_id = positive_ids[-4]
    article = Article.from_article(article_id, include_predictions=False)
    article.preprocess()
    print("id", str(article_id), "texcat", str(article.textcat_prediction))
    print(article.doc.text)
    print(json.dumps(article.get_investment_information_v1()["parsed_info"], indent=4))
