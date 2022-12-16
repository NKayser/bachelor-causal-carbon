import re

from MyCode.scripts.consts import MONEY_PATTERNS, IGNORE_MONEY_PATTERNS, TECHNOLOGY_CATEGORIES


def get_additional_money_ents(doc, money_ents):
    for pattern in MONEY_PATTERNS:
        matches = re.finditer(pattern, doc.text)
        count_matches = 0
        for match in matches:
            count_matches += 1
            match_text = doc.text[match.start():match.end()]
            if match_text in [ent.text for ent in money_ents]:
                continue
            match_in_existing = False
            for i in range(0, len(money_ents)):
                me_start = money_ents[i].start_char
                me_end = money_ents[i].end_char
                assert me_start < me_end and match.start() < match.end()
                if not (me_end <= match.start() or match.end() <= me_start):
                    money_ents[i] = doc.char_span(min(me_start, match.start()), max(me_end, match.end()), label="MONEY",
                                                  alignment_mode="expand")
                    match_in_existing = True
            if not match_in_existing:
                new_money_ent = doc.char_span(match.start(), match.end(), label="MONEY", alignment_mode="expand")
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


def get_weighted_technology_cats(article_text, categories=TECHNOLOGY_CATEGORIES):
    # initialize the counts for each category
    counts = {c: 0 for c in categories}

    # search for each keyword in the text
    for c, keywords in categories.items():
        for keyword in keywords:
            count = len(re.findall(keyword, article_text))
            counts[c] += count

    # print the counts for each category
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    sorted_counts = [x for x in sorted_counts if x[1] != 0]
    return sorted_counts