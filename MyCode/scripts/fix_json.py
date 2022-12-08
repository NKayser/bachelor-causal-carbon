from MyCode.scripts.consts import INPUT_PATH
from MyCode.scripts.utils import read_input_file, write_json_to_file

json_list = read_input_file()

for json_obj in json_list:
    if not (len(json_obj["label"]) > 0 and json_obj["dataset"] == "train"):
        write_json_to_file(json_obj, INPUT_PATH + "2")
    else:
        continue
