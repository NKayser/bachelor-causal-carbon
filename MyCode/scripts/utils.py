import json
import re
import socket
from collections import Counter
from functools import partial
from decimal import Decimal

from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from quantiphy import Quantity

from MyCode.scripts.consts import INPUT_PATH, MONEY_PATTERNS


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


def standardize_currency(cur):
    currency_dict = {"EUR": "EUR", "Eur": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR", "Euro": "EUR", "€": "EUR",
                     "USD": "USD", "Usd": "USD", "usd": "USD", "$": "USD", "US$": "USD", "us$": "USD", "Us$": "USD",
                     "CAD": "CAD", "CAN$": "CAD", "CAN": "CAD", "CAD$": "CAD", "cad": "CAD", "cad$": "CAD",
                     "can": "CAD", "can$": "CAD", "Cad": "CAD", "Cad$": "CAD", "Can": "CAD", "Can$": "CAD",
                     "CHF": "CHF", "Chf": "CHF", "chf": "CHF",
                     "PLN": "PLN", "pln": "PLN", "Pln": "PLN",
                     "GBP": "GBP", "\u00a3": "GBP", "pound": "GBP", "pounds": "GBP"}
    # TODO: add these GBP synonyms to MONEY_PATTERNS
    if cur in currency_dict.keys():
        return currency_dict[cur]
    return cur.upper()


def magnitude_str_to_number(mag):
    magnitude_dict = {"m": 1000000, "mio": 1000000, "mln": 1000000, "million": 1000000,
                      "b": 1000000000, "bn": 1000000000, "billion": 1000000000,
                      "thousand": 1000}
    if mag in magnitude_dict.keys():
        return Decimal(magnitude_dict[mag])
    return Decimal(1)


def number_str_to_decimal(num):
    if num is None or num == "":
        return None
    last_point_ind = num.rfind(".")
    last_comma_ind = num.rfind(",")

    if last_point_ind > last_comma_ind:
        out = num.replace(",", "")
        return Decimal(out)
    elif last_comma_ind > last_point_ind:
        out = num.replace(".", "").replace(",", ".")
        return Decimal(out)
    return Decimal(num)


def parse_location(loc_str):
    exceptions = {"U.S.": "US", "the United States": "US"}
    geolocator = Nominatim(user_agent="causal_carbon_bachelor")
    geocode = partial(geolocator.geocode, language='en')
    geocode = RateLimiter(geocode, min_delay_seconds=1)

    if loc_str in exceptions.keys():
        loc_str = exceptions[loc_str]
    try:
        coded = geocode(loc_str)
        name = coded.raw["display_name"]
        return {"parsed": name, "original": loc_str}
    except socket.timeout:
        print(loc_str + " not found")
        return {"parsed": None, "original": loc_str}
    except GeocoderUnavailable:
        print(loc_str + " not found")
        return {"parsed": None, "original": loc_str}


def parse_money(money_str):
    matches = [re.search(money_pat, money_str) for money_pat in MONEY_PATTERNS]

    if matches[0] is not None:
        groups = matches[0].groups()
        currency = groups[0]
        lower = groups[1]
        upper = groups[3]
        magnitude = groups[5]
        print(currency, lower, upper, magnitude)
    elif matches[1] is not None:
        groups = matches[1].groups()
        currency = groups[5]
        lower = groups[0]
        upper = groups[2]
        magnitude = groups[4]
        print(currency, lower, upper, magnitude)
    else:
        return None

    currency = standardize_currency(currency)
    magnitude = magnitude_str_to_number(magnitude)
    lower = number_str_to_decimal(lower)
    upper = number_str_to_decimal(upper)
    if lower:
        lower *= magnitude
    else:
        lower = upper
    if upper:
        upper *= magnitude
    else:
        upper = lower

    return {"currency": currency, "lower_value": str(lower), "upper_value": str(upper), "original": money_str}


def parse_percent(percent_str):
    try:
        match = re.search("([0-9]*[.,])?[0-9]+", percent_str)
        if match is not None:
            return {"value": str(Decimal(match[0].replace(",", "."))), "unit": "%", "original": percent_str}
        return {"value": None, "unit": "%", "original": percent_str}
    except:
        return {"value": None, "unit": "%", "original": percent_str}


def parse_quantity(quant_str):
    try:
        quant = Quantity(quant_str)
        value, unit = quant.as_tuple()
        return {"value": str(value), "unit": unit, "original": quant_str}
    except:
        return {"value": None,  "unit": None, "original": quant_str}


def parse_time(time_str):
    # left out for now
    return time_str




def sort_by_ent_cat(spans, ent_cats, threshold):
    # Create a list of tuples, where each tuple contains the span from `spans1` and the span from `spans2`
    scores = ent_cats.attrs["scores"]
    zipped_ent_cats = list(zip(ent_cats, scores))
    span_pairs = []
    for s1 in spans:
        found = False
        for s2, score in zipped_ent_cats:
            if s1.start == s2.start and s1.end == s2.end:
                found = True
                span_pairs.append((s1, s2, score))
                break
        if found:
            continue
        # not clear why some suggested spans don't appear in ent_cat list at all
        s2 = s1
        s2.label_ = "neutral"
        span_pairs.append((s1, s2, 1.0))
    # Sort the list of tuples by the category and confidence score of the span from `spans2`
    def order(label, score):
        out = 0
        if label == "positive":
            out += (0.5 + score / 2)
        elif label == "neutral":
            out += 0.5
        else:
            out += (0.5 - score / 2)
        return out

    span_pairs = [(s1.text, order(s2.label_, score.item())) for s1, s2, score in span_pairs]
    span_pairs.sort(key=lambda x: x[1], reverse=True)

    # filter by threshold confidence
    span_pairs = list(filter(lambda x: x[1] >= threshold, span_pairs))

    # deduplicate, keep highest confidence span
    temp = []
    for ent_text, score in span_pairs:
        if ent_text in temp:
            continue
        temp.append(ent_text)

    # Return the sorted list of spans from `spans1`
    return temp


def read_input_file(path=INPUT_PATH):
    with open(path, "r", encoding="utf8") as json_file:
        json_list = list(json_file)
        return [json.loads(json_str) for json_str in json_list]


def write_json_to_file(obj, path):
    with open(path, "a", encoding="utf8") as json_file:
        json_file.write(json.dumps(obj) + "\n")
