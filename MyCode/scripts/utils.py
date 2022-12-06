import json
import socket
from collections import Counter
from functools import partial

from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from MyCode.scripts.consts import INPUT_PATH

ALL_STANDARD_ENTITY_TYPES = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
                             "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]


def get_positive_article_ids():
    with open(INPUT_PATH, "r", encoding="utf-8") as input_file:
        json_list = list(input_file)

    positive_ids = []

    for json_str in json_list:
        article_data = json.loads(json_str)
        article_labels = article_data["label"]
        if article_labels != [] and article_labels[0] == "positive":
            positive_ids.append(article_data["id"])

    return positive_ids


def get_entities_of_type(text, all_entities, type):
    return sorted(dict(Counter([text[ent["start_offset"]:ent["end_offset"]] for ent in all_entities if ent["label"] == type])).items(), key=lambda k: k[1], reverse=True)


def get_all_entities_by_type(text, all_entities):
    all_types = list(dict.fromkeys([ent["label"] for ent in all_entities]))
    return {type: get_entities_of_type(text, all_entities, type) for type in all_types}


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
                        out[i][1] += s[1]
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
                removed_name = out.pop(i)
                removed_bounding_box = bounding_boxes.pop(i)
            j += 1
        i += 1

    out = list(dict.fromkeys(out))

    return out
