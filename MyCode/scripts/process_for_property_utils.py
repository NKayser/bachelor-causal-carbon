import re

from MyCode.scripts.Charspan import Charspan
from MyCode.scripts.consts import MONEY_PATTERNS, IGNORE_MONEY_PATTERNS


def get_additional_money_ents(text, money_ents):
    for pattern in MONEY_PATTERNS:
        matches = re.finditer(pattern, text)
        count_matches = 0
        for match in matches:
            count_matches += 1
            match_text = text[match.start():match.end()]
            if match_text in [ent.text for ent in money_ents]:
                continue
            match_in_existing = False
            for i in range(0, len(money_ents)):
                me_start = money_ents[i].start_offset
                me_end = money_ents[i].end_offset
                assert me_start < me_end and match.start() < match.end()
                if not (me_end <= match.start() or match.end() <= me_start):
                    money_ents[i].set_new_offset(min(me_start, match.start()), max(me_end, match.end()))
                    match_in_existing = True
            if not match_in_existing:
                new_money_ent = Charspan(match.start(), match.end(), "MONEY", text, "ent")
                money_ents.append(new_money_ent)

    i = 0
    while i < len(money_ents):
        ent = money_ents[i]
        i_popped = False
        for pattern in IGNORE_MONEY_PATTERNS:
            if re.fullmatch(pattern, ent.text):
                money_ents.pop(i)
                i_popped = True
        if not i_popped:
            i += 1

    return money_ents