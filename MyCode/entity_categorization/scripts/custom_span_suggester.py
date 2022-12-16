from collections import Counter
from typing import List, Optional, Iterable, cast

import tokenizers
from spacy.pipeline.spancat import Suggester
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d

from spacy.tokens import Doc
from spacy.util import registry
import spacy
import re


# assumes execution from entity_categorization
INPUT_PATH = "../data/labels_and_predictions.jsonl"
TEXTCAT_MODEL_PATH = "../textcat/models/cval_2/model-best"
SPANCAT_MODEL_PATH = "../sententence_categorization/models/model-best"
PRETRAINED_NER_MODEL = "en_core_web_trf"


TEXTCAT_THRESHOLD = 0.5

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

WEIGHTED_SENT_KEYWORDS = [("investment", 4),
                          ("invest", 3),
                          ("project", 2),
                          ("technology", 2),
                          ("plant", 2),
                          ("CO2", 2),
                          ("carbon", 2),
                          ("Carbon", 2),
                          ("environment", 1),
                          ("sustain", 1)] # with weights


def filter_ents(ents, label):
    return list(filter(lambda ent: ent.label == label, ents))


def get_entities_with_label(all_entities, label):
    return sorted(dict(Counter([ent for ent in all_entities if ent.label == label])).items(),
                  key=lambda k: k[1], reverse=True)


def ent_is_in_sent(ent, sent):
    # or overlapping
    return not (ent.end_offset < sent.start_offset or ent.start_offset > sent.end_offset)


def get_ents_of_sent(all_entities, sent):
    return [ent for ent in all_entities if ent_is_in_sent(ent, sent)]


def apply_spacy_model(input_text, model):
    spacy.prefer_gpu(0)
    nlp = spacy.load(model)
    doc = nlp(input_text)
    return doc


def apply_sentencizer(input_text, model):
    doc = apply_spacy_model(input_text, model)
    return [{"id": sent.ent_id, "start_offset": sent.start_char, "end_offset": sent.end_char}
            for sent in doc.sents]


def apply_textcat(input_text, model):
    doc = apply_spacy_model(input_text, model)
    return doc.cats["positive"]


def apply_spancat(input_text, model):
    doc = apply_spacy_model(input_text, model)
    formatted_spans = []
    running_span_id = 0

    for span in doc.spans['sc']:
        formatted_spans.append(
            {"id": running_span_id, "label": span.label_, "start_offset": span.start_char, "end_offset": span.end_char})
        running_span_id += 1

    return formatted_spans


def apply_pretrained_ner(input_text, model):
    doc = apply_spacy_model(input_text, model)
    formatted_ents = []
    running_ent_id = 0

    for ent in doc.ents:
        formatted_ents.append(
            {"id": running_ent_id, "label": ent.label_, "start_offset": ent.start_char, "end_offset": ent.end_char})
        running_ent_id += 1

    return formatted_ents


def get_additional_money_ents(text, money_ents):
    for pattern in MONEY_PATTERNS:
        matches = re.finditer(pattern, text)
        count_matches = 0
        for match in matches:
            count_matches += 1
            match_text = text[match.start():match.end()]
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
                new_money_ent = Charspan(match.start(), match.end(), "MONEY", text, "ent")
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

    def preprocess_spacy(self, textcat_model=TEXTCAT_MODEL_PATH, spancat_model=SPANCAT_MODEL_PATH,
                         ner_model=PRETRAINED_NER_MODEL):
        self.textcat_prediction = apply_textcat(self.text, textcat_model)
        self.sents = Charspan.from_dict_array(apply_sentencizer(self.text, spancat_model), self.text, "sent")
        self.spans = Charspan.from_dict_array(apply_spancat(self.text, spancat_model), self.text, "span")
        self.ents = Charspan.from_dict_array(apply_pretrained_ner(self.text, ner_model), self.text, "ent")

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

    def get_financial_information(self):
        money_ents = filter_ents(self.ents, "MONEY")
        money_ents = get_additional_money_ents(self.text, money_ents)
        return money_ents

    def get_technology_ents(self, categories=TECHNOLOGY_CATEGORIES):
        technology_ents = []
        for c, keywords in categories.items():
            for keyword in keywords:
                for match in re.finditer(keyword, self.text):
                    new_tech_ent = Charspan(match.start(), match.end(), "TECHWORD", self.text, "ent")
                    technology_ents.append(new_tech_ent)
        return technology_ents


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


@registry.misc("article_all_ent_suggester.v1")
def build_custom_suggester() -> Suggester:
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

            article = Article(metadata=None, text=doc.text)
            article.preprocess_spacy()
            finance_ents = article.get_financial_information()      # "MONEY"
            technology_ents = article.get_technology_ents()         # "TECHWORD"
            location_ents = filter_ents(article.ents, "GPE")        # "GPE"
            quantity_ents = filter_ents(article.ents, "QUANTITY")   # "QUANTITY"
            percent_ents = filter_ents(article.ents, "PERCENT")     # "PERCENT"
            date_ents = filter_ents(article.ents, "DATE")           # "DATE"
            fac_ents = filter_ents(article.ents, "FAC")             # "FAC"
            product_ents = filter_ents(article.ents, "PRODUCT")     # "PRODUCT"
            parsed_ents = [finance_ents, technology_ents, location_ents, quantity_ents, percent_ents, date_ents,
                           fac_ents, product_ents]

            for ent_arr in parsed_ents:
                for ent in ent_arr:
                    if (ent.start_offset, ent.end_offset) not in cache:
                        spans.append((ent.start_offset, ent.end_offset))
                        cache.add((ent.start_offset, ent.end_offset))
                        length += 1

            lengths.append(length)

        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

        return output

    return custom_suggester
