import json
import re
import socket
from collections import Counter
from functools import partial

from geopy.exc import GeocoderUnavailable
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from quantiphy import Quantity
from numerizer import numerize

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


def parse_weighted_tech(x):
    tech_cat, confidence = x
    return {"category": tech_cat, "confidence": confidence}


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
        return magnitude_dict[mag]
    return 1


def numerize_wrapper(num_str):
    num_str = numerize(num_str)
    last_point_ind = num_str.rfind(".")
    last_comma_ind = num_str.rfind(",")
    number_index_list = list(re.finditer(r'\d', num_str))
    if number_index_list is None or len(number_index_list) == 0:
        last_number_index = None
    else:
        last_number_index = number_index_list[-1].start()

    if last_point_ind == -1:
        if last_comma_ind != -1 and last_number_index is not None and last_comma_ind == last_number_index - 3:
            out = num_str.replace(",", "")
            return out
        out = num_str.replace(",", ".")
        return out
    if last_comma_ind == -1:
        if last_point_ind != -1 and last_number_index is not None and last_point_ind == last_number_index - 3:
            out = num_str.replace(".", "")
            return out
        return num_str

    if last_point_ind > last_comma_ind:
        out = num_str.replace(",", "")
        return out
    elif last_comma_ind > last_point_ind:
        out = num_str.replace(".", "").replace(",", ".")
        return out
    return num_str


def number_str_to_decimal(num):
    if num is None or num == "":
        return None
    try:
        return float(numerize_wrapper(num))
    except:
        return None


def parse_location(x):
    # TODO: weird location for United Kingdom
    original, confidence = x
    loc_str = original
    exceptions = {"U.S.": "US", "the United States": "US"}
    geolocator = Nominatim(user_agent="causal_carbon_bachelor")
    geocode = partial(geolocator.geocode, language='en')
    geocode = RateLimiter(geocode, min_delay_seconds=1)

    if loc_str in exceptions.keys():
        loc_str = exceptions[loc_str]
    try:
        coded = geocode(loc_str)
        name = coded.raw["display_name"]
        return {"parsed": name, "original": original, "confidence": confidence}
    except AttributeError:
        return {"parsed": None, "original": original, "confidence": confidence}
    except socket.timeout:
        print(loc_str + " not found")
        return {"parsed": None, "original": original, "confidence": confidence}
    except GeocoderUnavailable:
        print(loc_str + " not found")
        return {"parsed": None, "original": original, "confidence": confidence}


def parse_money(x):
    # TODO: "$30.15" not parsed in 6019. Might also exclude unsuccessfully parsed ents
    # And "EUR 26.4227", "USD 37.8682"
    # "U.S.$625,630,000" in 6454
    # Numbers smaller than 100000 can be ignored anyways
    # "C$15.5 million" in 6080 should be CAD, not USD
    # Also: still provide value even if currency not found?
    # look for keywords "over", "more", "at least", "greater" -> upper should be null
    # new currencies: GBP, INR, Rs., CZK
    money_str, confidence = x
    matches = [re.findall(money_pat, money_str) for money_pat in MONEY_PATTERNS]
    magnitude2 = None

    if matches[0] is not None and len(matches[0]) > 0:
        currency = matches[0][0][0]
        lower = matches[0][0][1]
        upper = matches[0][0][3]
        magnitude = matches[0][0][5]
        if len(matches[0]) >= 2:
            currency2 = matches[0][1][0]
            if currency2 == currency:
                magnitude2 = matches[0][1][5]
                upper = matches[0][1][1]
    elif matches[1] is not None and len(matches[1]) > 0:
        currency = matches[1][0][5]
        lower = matches[1][0][0]
        upper = matches[1][0][2]
        magnitude = matches[1][0][4]
        if len(matches[1]) >= 2:
            currency2 = matches[1][1][5]
            if currency2 == currency:
                magnitude2 = matches[1][1][4]
                upper = matches[1][1][0]
    else:
        return {"currency": None, "lower_value": None, "upper_value": None, "original": money_str,
            "confidence": confidence}

    currency = standardize_currency(currency)
    magnitude = magnitude_str_to_number(magnitude)
    lower = number_str_to_decimal(lower)
    upper = number_str_to_decimal(upper)
    if lower:
        lower *= magnitude
    else:
        lower = upper
    if upper:
        if magnitude2 is not None:
            magnitude2 = magnitude_str_to_number(magnitude2)
            upper *= magnitude2
        else:
            upper *= magnitude
    else:
        upper = lower

    return {"currency": currency, "lower_value": lower, "upper_value": upper, "original": money_str,
            "confidence": confidence}


def parse_percent(x):
    percent_str, confidence = x
    percent_str = numerize_wrapper(percent_str)
    try:
        match = re.search("([0-9]*[.,])?[0-9]+", percent_str)
        if match is not None:
            return {"value": float(match[0].replace(",", ".")), "unit": "%", "original": percent_str,
                    "confidence": confidence}
        return {"value": None, "unit": "%", "original": percent_str, "confidence": confidence}
    except:
        return {"value": None, "unit": "%", "original": percent_str, "confidence": confidence}


def parse_quantity(x):
    # TODO: "630k MT" in 6037 -> megaton?
    # ton -> t ?
    # better handling of square and cubic
    # new field for type of physical quantity (weight, distance, area, volume)
    input_text, confidence = x
    quant_str = input_text
    replacements = {"kilograms": "kg", "kilogram": "kg", "megawatts": "MW", "megawatt": "MW", "-": " ",
                    "metre": "meter", "net": "",
                    "gigawatts": "GW", "gigawatt": "GW", "hour": "h", "tonnes": "t", "tonne": "t",
                    "tons": "t", "ton": "t", "meters": "m", "meter": "m", "kilometers": "km", "kilometer": "km",
                    "metric tons": "t", "metric": "", "cubic": "", "square": "",
                    "feet": "ft", "foot": "ft", "pounds": "lb", "pound": "lb", "oil-equivalent-barrel": "BOE",
                    "kilo": "k"}
    for key, value in replacements.items():
        quant_str = quant_str.replace(key, value)
    quant_str = numerize_wrapper(quant_str)
    try:
        new_quant_str = re.split(r'(^[^\d]+)', quant_str)[-1]
        quant = Quantity(new_quant_str)
        value, unit = quant.as_tuple()
        return {"value": value, "unit": unit, "original": input_text, "confidence": confidence}
    except:
        try:
            new_quant_str = re.split(r'(^[^\d]+)', quant_str)[-1]
            new_quant_str = new_quant_str.split()
            quant = Quantity(new_quant_str[0] + " " + new_quant_str[1])
            value, unit = quant.as_tuple()
            return {"value": value, "unit": unit, "original": input_text, "confidence": confidence}
        except:
            return {"value": None,  "unit": None, "original": input_text, "confidence": confidence}


def parse_time(x):
    # TODO: could try parsing some standard date formats
    time_str, confidence = x
    # left out for now
    return {"original": time_str, "confidence": confidence}


def sort_by_ent_cat(spans, ent_cats, threshold=-1.0):
    if len(ent_cats) == 0:
        return []
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
        span_pairs.append((s1, s2, 0.0))
    # Sort the list of tuples by the category and confidence score of the span from `spans2`
    def order(label, score):
        out = 0
        if label == "positive":
            out += score
        elif label == "neutral":
            out += 0.0
        else:
            out -= score
        return out

    def convert_score(score):
        try:
            score = score.item()
            return score
        except AttributeError:
            return score

    span_pairs = [(s1.text, order(s2.label_, convert_score(score))) for s1, s2, score in span_pairs]
    span_pairs.sort(key=lambda x: x[1], reverse=True)

    # filter by threshold confidence
    span_pairs = list(filter(lambda x: x[1] >= threshold, span_pairs))

    # deduplicate, keep highest confidence span
    temp = []
    for ent_text, score in span_pairs:
        cont = False
        if spans[0] is not None and spans[0].label_ == "QUANTITY":
            for ignore_keyword in ["degree", "mile", "acre"]: #  "meter", "feet"
                if ignore_keyword in ent_text:
                    cont = True
                    break
        if cont:
            continue
        # TODO: some duplicates still get through, like "US" in 6008
        for t, s in temp:
            if t in ent_text:
                temp.remove((t, s))
                break
            if ent_text in t:
                cont = True
                break
        if cont:
            continue
        temp.append((ent_text, score))

    # Return the sorted list of spans from `spans1`
    return temp


def read_input_file(path=INPUT_PATH):
    with open(path, "r", encoding="utf8") as json_file:
        json_list = list(json_file)
        return [json.loads(json_str) for json_str in json_list]


def write_json_to_file(obj, path):
    with open(path, "a", encoding="utf8") as json_file:
        json_file.write(json.dumps(obj) + "\n")
