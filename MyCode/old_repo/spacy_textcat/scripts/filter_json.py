import json

with open("../assets/textcat_all.jsonl", 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

with open("../assets/textcat_only_positive.jsonl", 'w', encoding='utf-8') as out_file:
    for json_str in json_list:
        result = json.loads(json_str)
        if len(result["label"]) > 0 and result["label"][0] == "positive":
            #print(str(result["id"]) + " " + result["text"][0:100])
            out_file.write(json_str)