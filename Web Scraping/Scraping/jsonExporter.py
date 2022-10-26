import pandas as pd

import GeneralNoaDB.DBTableNames as tb
from GeneralInfrastructure.Database.DatabaseManager import dbm, Database

from webScrapingV1 import getTableNames


def getDfFromDb():
    noadb = dbm.get_database(Database.NOADB)
    td = getTableNames()
    pages_in_tbl = td['pagetracker_nk_pages_in']
    pages_out_tbl = td['pagetracker_nk_out']

    sql = f"""SELECT * FROM {pages_out_tbl} WHERE language = 'English'"""# AND carbon_in_title = 1 AND carbon_in_text = 1"""
    df = noadb.get_as_df(sql)
    return df


def saveDfAsJson():
    df = getDfFromDb()
    out = df.to_json(path_or_buf="../JsonExport/all_data_unlabeled.json", lines=True, orient="records", date_format="iso")


def jsonToDb():
    noadb = dbm.get_database(Database.NOADB)
    td = getTableNames()
    pages_out_tbl = td['pagetracker_nk_out']

    df = pd.read_json("../JsonExport/annotated_relations.jsonl", lines=True, orient="records")
    noadb.insert_df(pages_out_tbl, df)


if __name__ == '__main__':
    #saveDfAsJson()
    jsonToDb()