import spacy


def apply_spacy_model(input_text, model):
    spacy.prefer_gpu(0)
    nlp = spacy.load(model)
    doc = nlp(input_text)
    return doc


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
