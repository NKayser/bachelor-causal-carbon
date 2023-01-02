import json
import socket
from collections import Counter
from functools import partial

from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from MyCode.scripts.consts import INPUT_PATH


def get_positive_article_ids(path=INPUT_PATH):
    with open(path, "r", encoding="utf-8") as input_file:
        json_list = list(input_file)

    positive_ids = []

    for json_str in json_list:
        article_data = json.loads(json_str)
        article_labels = article_data["label"]
        if article_labels != [] and article_labels[0] == "positive":
            positive_ids.append(article_data["id"])

    return positive_ids


def get_entities_with_label(all_entities, label):
    return sorted(dict(Counter([ent for ent in all_entities if ent.label_ == label])).items(),
                  key=lambda k: k[1], reverse=True)


def get_all_entities_by_label(all_entities):
    all_labels = list(dict.fromkeys([ent.label_ for ent in all_entities]))
    return {label: get_entities_with_label(all_entities, label) for label in all_labels}


def ent_is_in_sent(ent, sent):
    # or overlapping
    return not (ent.end_char < sent.start_char or ent.start_char > sent.end_char)


def get_ents_of_sent(all_entities, sent):
    return [ent for ent in all_entities if ent_is_in_sent(ent, sent)]


def get_span_labels_of_sentence(spans, sent):
    return [span.label for span in spans if span.start_char == sent.start_char]


def filter_ents(ents, label):
    return list(filter(lambda ent: ent.label_ == label, ents))


def opposite_filter_ents(ents, label):
    return list(filter(lambda ent: ent.label_ != label, ents))


def ent_to_token_slice(doc, ent):
    span = doc.char_span(ent.start_char, ent.end_char)
    return span[0].i, span[-1].i + 1


def rectangle_subset(rect1, rect2):
    return rect1[0] > rect2[0] and rect1[1] < rect2[1] and rect1[2] > rect2[2] and rect1[3] < rect2[3]


def get_more_precise_locations(loc_array):
    exceptions = {"U.S.": "US", "the United States": "US"}

    geolocator = Nominatim(user_agent="causal_carbon_bachelor")
    geocode = partial(geolocator.geocode, language='en')
    geocode = RateLimiter(geocode, min_delay_seconds=1)

    locations = []
    out = []
    bounding_boxes = []

    for s in loc_array:
        input_name = s[0]
        if input_name in exceptions.keys():
            input_name = exceptions[input_name]
        try:
            coded = geocode(input_name)
            name = coded.raw["display_name"]
            if name not in locations:
                locations.append(name)
                #out.append((name, input_name, s[1]))
                out.append((name, s[1]))
                bounding_boxes.append(coded.raw["boundingbox"])
            else:
                # add counts of duplicate to first mention of location
                for i in range(0, len(out)):
                    if out[i][0] == name:
                        out[i] = (out[i][0], out[i][1] + s[1])
        except socket.timeout:
            print(s[0] + " not found")
            continue
        except GeocoderUnavailable:
            print(s[0] + " not found")
            continue

    # remove countries if already mentioned in more specific place
    i = 0
    while i < len(out):
        j = 0
        while j < len(out):
            if i != j and (out[i][0] in out[j][0]): # or rectangle_subset(bounding_boxes[j], bounding_boxes[i])):
                out[j][1] += out[i][1]
                removed_name = out.pop(i)
                removed_bounding_box = bounding_boxes.pop(i)
            j += 1
        i += 1

    #out = list(dict.fromkeys(out))

    return out


def sort_by_ent_cat(spans, ent_cats):
    # Create a list of tuples, where each tuple contains the span from `spans1` and the span from `spans2`
    span_pairs = []
    for s1 in spans:
        for s2 in ent_cats:
            if s1.start == s2.start and s1.end == s2.end:
                span_pairs.append((s1, s2))
                break
    # Sort the list of tuples by the category and confidence score of the span from `spans2`
    span_pairs.sort(key=lambda x: (x[1].label_ == "positive", -x[1].score, x[1].label_ == "negative", x[1].score))
    print(span_pairs)
    # Return the sorted list of spans from `spans1`
    return [s[0] for s in span_pairs]


def read_input_file(path=INPUT_PATH):
    with open(path, "r", encoding="utf8") as json_file:
        json_list = list(json_file)
        return [json.loads(json_str) for json_str in json_list]


def write_json_to_file(obj, path):
    with open(path, "a", encoding="utf8") as json_file:
        json_file.write(json.dumps(obj) + "\n")
