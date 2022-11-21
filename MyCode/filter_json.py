import json

with open("admin.jsonl", 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    if result["label"][0] == "unsure":
        print(str(result["id"]) + " " + result["text"][0:100])