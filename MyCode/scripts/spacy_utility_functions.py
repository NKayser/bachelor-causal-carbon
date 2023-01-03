import spacy
from spacy import registry

from MyCode.entity_categorization.scripts.custom_span_suggester import build_custom_suggester


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



@registry.misc("article_all_ent_suggester.v1")
def suggester():
    return build_custom_suggester(balance=False, input_path="data/labels_and_predictions.jsonl")