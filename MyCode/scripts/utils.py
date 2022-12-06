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


def get_all_entities_by_type(text, all_entities, all_types=ALL_STANDARD_ENTITY_TYPES):
    return {type: get_entities_of_type(text, all_entities, type) for type in all_types}


def get_more_precise_locations(loc_array):
    geolocator = Nominatim(user_agent="causal_carbon_bachelor")
    geocode = partial(geolocator.geocode, language='en')
    geocode = RateLimiter(geocode, min_delay_seconds=1)

    locations = []
    out = []

    for s in loc_array:
        try:
            coded = geocode(s[0])
            name = coded.raw["display_name"]
            print(s[0] + " -> " + name)
            if name not in locations:
                locations.append(name)
                out.append((name, s[1]))
        except socket.timeout:
            print(s[0] + " not found")
            continue
        except GeocoderUnavailable:
            print(s[0] + " not found")
            continue

    out = list(dict.fromkeys(out))

    return out
