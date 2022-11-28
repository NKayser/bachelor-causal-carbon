import json
import spacy

nlp = spacy.load("en_core_web_trf")
spacy.prefer_gpu(0)

with open("../../data/annotated_relations.jsonl", 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)

        doc = nlp(result["text"])

        #print("Entities for doc " + str(result["id"]))

        for ent in doc.ents:
            if ent.label_ == "GPE":
                print(ent.text, "|", result["text"][ent.start_char-20:ent.end_char+20], ent.label_)



# Spacy entities

# PERSON - People, including fictional.
# NORP - Nationalities or religious or political groups.
    # kind of related to location
# FAC - Buildings, airports, highways, bridges, etc.
    # useful for recognizing plant names, project names, sometimes even technology (DRI, Electric Arc Furnace, Blast Furnace) or location (Port-Cartier)
# ORG - Companies, agencies, institutions, etc.
    # works, but too many. Might still be useful to attach, if list deduplicated
# GPE - Countries, cities, states.
    # Perfect for location. Plugin that looks up country of a city?
# LOC - Non-GPE locations, mountain ranges, bodies of water.
    # Not very often useful. Mostly continents
# PRODUCT - Objects, vehicles, foods, etc. (Not services.)
    # What companies call there emissions-reduced products. Sometimes also project name?
# EVENT - Named hurricanes, battles, wars, sports events, etc.
    # if event mentioned lower likelihood that textcat positive? otherwise not useful
# WORK_OF_ART - Titles of books, songs, etc.
    # lots of false positives, not useful anyway
# LAW - Named documents made into laws.
    # actually useful for knowing which legislation is relevent here. e.g. Green DEAL, CBA, Paris Agreement, Scope 1 (for some reason)
# LANGUAGE - Any named language.
# DATE - Absolute or relative dates or periods.
    # relevant
# TIME - Times smaller than a day.
# PERCENT - Percentage, including "%".
    # not all relevant, but all accurate. Classify further into emissions related, goal/concrete, past/future?
# MONEY - Monetary values, including unit.
    # good, but things like €131m not recognized correctly (131, m). Also classify if relevant to investment
# QUANTITY - Measurements, as of weight or distance.
    # recognizes tonnes, for CO2, but also others of course
# ORDINAL - "first", "second", etc.
# CARDINAL - Numerals that do not fall under another type.