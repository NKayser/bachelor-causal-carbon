import json
#from collections import Counter
from typing import List, Optional, Iterable, cast

#import tokenizers
from spacy.pipeline.spancat import Suggester
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d

from spacy.tokens import Doc, SpanGroup
from spacy.util import registry
import spacy
import re


# assumes execution from entity_categorization
INPUT_PATH = "../data/labels_and_predictions.jsonl"
SPANCAT_MODEL_PATH = "../sententence_categorization/models/model-best"
PRETRAINED_NER_MODEL = "en_core_web_trf"

TECHNOLOGY_CATEGORIES = {
            'DRI': ['DRI', 'direct reduction', 'iron ore'],
            'EAF': ['EAF', 'Electric Arc Furnace'],
            'solar': ['solar'],
            'wind': ['wind', 'park', 'offshore'],
            'hydrogen': ['hydrogen'],
            'renewable': ['renewable energy', 'renewables', 'wind', 'solar', 'hydropower'],
            'CCGT': ['CCGT'],
            'TRT': ['TRT'],
            'HBI': ['HBI'],
            'carbon capture': ['CCUS', 'CCS', 'CCU', 'carbon capture', 'storage', 'utilization', 'store', 're-use'],
            'decommission coal plant': ['Shut down', 'shut down', 'decommission', 'coal'], # 'coal' only in combination relevant
            'plant efficiency': ['modern', 'Modern', 'heat', 'efficient', 'efficiency', 'plant', 'energy-saving',
                                 'recovery', 'production process'], # 'recovery' in combination with 'heat'
            'dust filter': [' dust', 'filter', 'sinter', 'Nm', 'smoke'], # 'industrial' not 'dust'
            'R&D': ['R&D', 'research and development', 'research'],
            'alternative fuel': ['alternative fuel', 'bio-fuel'],
            'recycling': ['recycling', 'recycle', 'recyclability', 're-use', 'scrap', 'by-product', 'waste'],
            'steel': ['steel', 'blast furnace', 'coke'],
            'cement': ['cement'],
            'microbe': ['microbe', 'microbial', 'bioethanol'],
            'general': ['environment', 'sustain', 'net zero', 'net-zero', 'neutral'],
        }

ALL_STANDARD_ENTITY_TYPES = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
                             "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]

MONEY_PATTERNS = ["(EUR|Eur|eur|euro|euros|Euro|€|\u20ac|USD|Usd|usd|\$|US\$|us\$|Us\$|CAN|CAN\$|CAD|CAD\$|cad|cad\$|can|can\$|Cad|Cad\$|Can|Can\$|CHF|Chf|chf|PLN|pln|Pln|\u00a3) ?(\d+([\.,]?\d*)*)[-–]?(\d*([\.,]?\d*)*)\+? ?(million|mio|mln|m|billion|bn|b|thousand)",
                       "(\d+([\.,]?\d*)*)[-–]?(\d*([\.,]?\d*)*)\+? ?(m|mio|mln|million|b|bn|billion|thousand| )\.? ?(EUR|Eur|eur|euro|euros|Euro|€|\u20ac|USD|Usd|usd|\$|US\$|us\$|Us\$|CAN|CAN\$|CAD|CAD\$|cad|cad\$|can|can\$|Cad|Cad\$|Can|Can\$|CHF|Chf|chf|PLN|pln|Pln|\u00a3)"]
IGNORE_MONEY_PATTERNS = ["2 can", "\+[\d ?\-?\-?]*", "19 Eur"]

corresponding_labels = {
    "GPE": ["location"],
    "TECHWORD": ["technology", "core reference"],
    "FAC": ["technology", "core reference"],
    "PRODUCT": ["technology", "core reference"],
    "PERCENT": ["emissions"],
    "QUANTITY": ["emissions"],
    "DATE": ["timeline"],
    "MONEY": ["financial information"]
}



def filter_ents(ents, label):
    return list(filter(lambda ent: ent.label_ == label, ents))


def opposite_filter_ents(ents, label):
    return list(filter(lambda ent: ent.label_ != label, ents))


def ent_to_token_slice(doc, ent):
    span = doc.char_span(ent.start_char, ent.end_char, alignment_mode="expand")
    return span[0].i, span[-1].i + 1


def ent_is_in_sent(ent, sent):
    # or overlapping
    return not (ent.end_char < sent.start_char or ent.start_char > sent.end_char)


def get_additional_money_ents(doc, money_ents):
    for pattern in MONEY_PATTERNS:
        matches = re.finditer(pattern, doc.text)
        count_matches = 0
        for match in matches:
            count_matches += 1
            match_text = doc.text[match.start():match.end()]
            if match_text in [ent.text for ent in money_ents]:
                continue
            match_in_existing = False
            for i in range(0, len(money_ents)):
                me_start = money_ents[i].start_char
                me_end = money_ents[i].end_char
                assert me_start < me_end and match.start() < match.end()
                if not (me_end <= match.start() or match.end() <= me_start):
                    money_ents[i] = doc.char_span(min(me_start, match.start()), max(me_end, match.end()), label="MONEY", alignment_mode="expand")
                    match_in_existing = True
            if not match_in_existing:
                new_money_ent = doc.char_span(match.start(), match.end(), label="MONEY", alignment_mode="expand")
                money_ents.append(new_money_ent)

    i = 0
    while i < len(money_ents):
        ent = money_ents[i]
        i_popped = False
        for pattern in IGNORE_MONEY_PATTERNS:
            if re.fullmatch(pattern, ent.text):
                money_ents.pop(i)
                i_popped = True
        if not i_popped:
            i += 1

    return money_ents


def read_input_file(path=INPUT_PATH):
    with open(path, "r", encoding="utf8") as json_file:
        json_list = list(json_file)
        return [json.loads(json_str) for json_str in json_list]


class Article:
    doc = None
    labeled_entities = None

    def __init__(self, text, model=PRETRAINED_NER_MODEL):
        assert text is not None
        spacy.prefer_gpu(0)
        nlp = spacy.blank("en")
        self.doc = nlp(text)
        ner = spacy.load(model)
        self.doc.spans["sc"] = [self.doc.char_span(ent.start_char, ent.end_char, ent.label_) for ent in ner(text).ents]
        for json_obj in read_input_file():
            if json_obj["text"][:200] == text[:200]:
                self.doc.spans["sc"] = [self.doc.char_span(span["start_offset"], span["end_offset"], span["label"])
                                        for span in json_obj["entities"]]
                self.labeled_entities = json_obj["labeled_entities"]
                break


    def get_financial_information(self):
        money_ents = filter_ents(self.doc.spans["sc"], "MONEY")
        money_ents = get_additional_money_ents(self.doc, money_ents)
        self.doc.spans["sc"] = opposite_filter_ents(self.doc.spans["sc"], "MONEY") + money_ents
        return money_ents

    def get_technology_ents(self, categories=TECHNOLOGY_CATEGORIES):
        technology_ents = SpanGroup(self.doc)
        for c, keywords in categories.items():
            for keyword in keywords:
                for match in re.finditer(keyword, self.doc.text):
                    new_tech_ent = self.doc.char_span(match.start(), match.end(), label="TECHWORD", alignment_mode="expand")
                    technology_ents.append(new_tech_ent)
        self.doc.spans["sc"] = self.doc.spans["sc"] + technology_ents
        return technology_ents


@registry.misc("article_all_ent_suggester.v1")
def build_custom_suggester(balance: bool = True) -> Suggester:
    """Suggest all spans of the given lengths. Spans are returned as a ragged
    array of integers. The array has two columns, indicating the start and end
    position."""

    def custom_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []

        for doc in docs:
            cache = set()
            length = 0
            doc_dist = [0, 0]

            article = Article(text=doc.text)
            finance_ents = article.get_financial_information()      # "MONEY"
            #technology_ents = article.get_technology_ents()         # "TECHWORD"
            #article_ents = article.doc.spans["sc"]
            #location_ents = filter_ents(article_ents, "GPE")        # "GPE"
            #quantity_ents = filter_ents(article_ents, "QUANTITY")   # "QUANTITY"
            #percent_ents = filter_ents(article_ents, "PERCENT")     # "PERCENT"
            #date_ents = filter_ents(article_ents, "DATE")           # "DATE"
            #fac_ents = filter_ents(article_ents, "FAC")             # "FAC"
            #product_ents = filter_ents(article_ents, "PRODUCT")     # "PRODUCT"
            #parsed_ents = [finance_ents, location_ents, quantity_ents, percent_ents, date_ents,
            #               fac_ents, product_ents] # technology_ents
            #parsed_ents = [finance_ents]

            relevant_labels = set()
            for arr in corresponding_labels.values():
                for val in arr:
                    relevant_labels.add(val)

            ents = article.doc.spans["sc"]

            for ent in ents:
                if ent.label_ not in corresponding_labels.keys():
                    continue
                ent_in_labeled_ent = False
                if article.labeled_entities is not None:
                    for labeled_entity in article.labeled_entities:
                        label = labeled_entity["label"]
                        if label not in relevant_labels:
                            continue
                        labeled_ent = doc.char_span(labeled_entity["start_offset"], labeled_entity["end_offset"], label,
                                                    alignment_mode="expand")
                        if label in corresponding_labels[ent.label_]:
                            if ent_is_in_sent(ent, labeled_ent):
                                #print(label, ent.label_, labeled_ent.text)
                                ent_in_labeled_ent = True
                                doc_dist[0] += 1
                                new_ent = doc.char_span(ent.start_char, ent.end_char, ent.label_,
                                                        alignment_mode="expand")
                                assert str(new_ent) != "None"
                                #print(new_ent)
                                token_slice = ent_to_token_slice(doc, new_ent)
                                if token_slice not in cache:
                                    spans.append(token_slice)
                                    cache.add(token_slice)
                                    length += 1
                if ent_in_labeled_ent:
                    continue
                if balance and doc_dist[1] > doc_dist[0]:
                    continue
                doc_dist[1] += 1
                new_ent = doc.char_span(ent.start_char, ent.end_char, ent.label_, alignment_mode="expand")
                assert str(new_ent) != "None"
                #print(new_ent)
                token_slice = ent_to_token_slice(doc, ent)
                if token_slice not in cache:
                    spans.append(token_slice)
                    cache.add(token_slice)
                    length += 1

            #for ent in ents:
            #    token_slice = ent_to_token_slice(doc, ent)
            #    if token_slice not in cache:
            #        spans.append(token_slice)
            #        cache.add(token_slice)
            #        length += 1

            lengths.append(length)

        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

        return output

    return custom_suggester
