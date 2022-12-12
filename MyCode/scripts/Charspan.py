class Charspan:
    id = None
    start_offset = None
    end_offset = None
    type = None
    label = None
    text = None
    article_text = None

    def __init__(self, start_offset, end_offset, label, article_text, span_type, span_id=None):
        assert article_text is not None
        self.article_text = article_text
        self.set_new_offset(start_offset, end_offset)
        self.id = span_id
        self.type = span_type
        self.label = label

    @classmethod
    def from_dict(cls, input_dict, article_text, span_type):
        if "label" in input_dict.keys():
            label = input_dict["label"]
        else:
            label = None
        return cls(input_dict["start_offset"], input_dict["end_offset"], label, article_text,
                   span_type, input_dict["id"])

    @classmethod
    def from_dict_array(cls, arr, article_text, span_type):
        return [cls.from_dict(span_dict, article_text, span_type) for span_dict in arr]

    def get_text(self, padding=0):
        start_char = self.start_offset - padding
        end_char = self.end_offset + padding
        if start_char < 0:
            start_char = 0
        if end_char > len(self.article_text):
            end_char = len(self.article_text)
        return self.article_text[start_char:end_char]

    def __str__(self):
        return str({"label": self.label, "text": self.text})

    def set_new_offset(self, new_start_offset, new_end_offset):
        if new_start_offset < 0 or new_end_offset > len(self.article_text):
            print("Entity boundaries outside of text")
            assert False
        self.start_offset = new_start_offset
        self.end_offset = new_end_offset
        self.text = self.get_text()