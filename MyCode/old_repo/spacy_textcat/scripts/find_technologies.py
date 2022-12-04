import json

technologies = ["DRI", "EAF", "Electric Arc Furnace", "solar", "wind", "CCGT", "hydrogen", "TRT", "CCUS", "CCS",
                "carbon capture", "TRT", "HBI", "electrolysis", "R&D", "research and development", "filter", "dust",
                "renewable energy", "scrap", "recycling", "coal", "activated clay", "heat", "kiln", "modern", "Modern",
                "alternative fuel", "energy", "Shut down", "shut down", "decommission"]
occurence = dict((k, 0) for k in technologies)
num_found = [0, 0]
unclassified = []
articles = dict()

with open("../assets/textcat_only_positive.jsonl", 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    found_a_tech = False
    techs_found = []

    for tech in technologies:
        if tech in result["text"]:
            occurence[tech] += 1
            techs_found.append(tech)

    if len(techs_found) > 0:
        num_found[0] += 1
    else:
        num_found[1] += 1
        unclassified.append(result["id"])

    articles[result["id"]] = techs_found

print(str(articles).replace("], ", "],\n"))
print("\n")
print(occurence)
print("Articles classified: " + str(num_found[0]) + ", not classified: " + str(num_found[1]))
print("Unclassified ids: " + str(unclassified))