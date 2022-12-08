import spacy

from MyCode.scripts.consts import INPUT_PATH
from MyCode.scripts.utils import read_input_file, write_json_to_file

nlp = spacy.load("../spancat/models/model-best")
json_list = read_input_file()

for json_obj in json_list:
    doc = nlp(json_obj["text"])
    json_obj["sents"] = [{"id": sent.ent_id, "start_offset": sent.start_char, "end_offset": sent.end_char}
                         for sent in doc.sents]
    write_json_to_file(json_obj, INPUT_PATH + "2")
