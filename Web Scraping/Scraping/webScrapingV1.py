import re
from datetime import datetime
from functools import reduce

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import requests
import pandas as pd
import cloudscraper as cs
import httpx
import html2text

import GeneralNoaDB.DBTableNames as tb
from GeneralInfrastructure.Database.DatabaseManager import dbm, Database


def getTableNames():
    table_dict = {
        'pagetracker_nk_out': tb.pagetracker_nk_out(tb.UNSYNCED),
        'pagetracker_nk_pages_in': tb.pagetracker_nk_pages_in(tb.UNSYNCED)
    }
    return table_dict


def getFunctionDict():
    function_dict = {
        1: getRweNews,
        2: getEphNews,
        3: getPgeNews,
        4: getArcelorNews,
        5: getEngieNews,
        6: getHeidelbergNews,

        9: getTotalNews,

        11: getHolcimNews,

        13: getThyssenNews,

        15: getOrlenNews,

        18: getExxonNews,

        23: getBasfNews,
    }
    return function_dict


def getRweNews(soup, page, max_articles=1000):
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0'})
    s.cookies.update({'gdpr-settings-cookie': '1|true|16.8.2022|16.8.2024|1059955080'})

    base_link = "https://www.rwe.com"

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all("a", attrs={"class": "results-links"})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 20 == 0:
            print("RWE article " + str(len(df_company.index)))
        error_text = ""

        # gather headline information
        title = item.find("h3")
        if title is None:
            count_errors += 1
            error_text += "h3 not found. "
            continue
        else:
            title = title.contents[0]
        if title in df_company.values:
            continue
        date_str = item.find("span", attrs={"class": "date"}).contents[0].strip()
        published_on = datetime.strptime(date_str, '%d.%m.%Y').date()
        article_url = item['href']
        article_category = item.find("div", attrs={"class": "target-group"}).contents[0]

        # go to article page
        page = s.get(article_url).content
        sub_soup = BeautifulSoup(page, 'html.parser')

        # get pr pdf
        pdf = sub_soup.find("a", attrs={"aria-label": "Pressemitteilung"})
        successful = False
        if pdf is not None:
            try:
                pdf_url = base_link + pdf['href']
                successful = True
            except:
                pdf_url = None

        if not successful:
            pdf_url = None
            error_text += "pdf_url not found. "
            count_errors = count_errors + 1

        # get article text
        try:
            relevant_part_str = sub_soup.prettify().split('<h2 class="headline">')[1].split('<div data-tpl="con01">')[0]
            relevant_part = BeautifulSoup(relevant_part_str, 'html.parser')
        except:
            relevant_part = sub_soup
            error_text += "Relevant part not determined. "
            print("RWE: Error with getting relevant part")
        text_list = relevant_part.findAll(text=True)
        text_list = [value.strip() for value in text_list if value != "\n" and "[if IE 9]" not in value]
        article_text = "\n".join(text_list)

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'article_url', 'article_category',
                                               'article_text', 'pdf_url', 'error_text', 'language']] =\
            [published_on, title, article_url, article_category, article_text, pdf_url, error_text, "German"]

        if len(df_company.index) >= max_articles:
            print("Gathered RWE headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered RWE headlines with " + str(count_errors) + " error(s) and " + str(count_warnings) + " warning(s).")
    return df_company


def getRweNewsEnglish(soup, page, max_articles=1000):
    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0'})
    s.cookies.update({'gdpr-settings-cookie': '1|true|16.8.2022|16.8.2024|1059955080'})

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.body_width = 0
    h.ignore_images = True
    h.ignore_tables = True

    base_link = "https://www.rwe.com"

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all("a", attrs={"class": "results-links"})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 20 == 0:
            print("RWE article " + str(len(df_company.index)))
        error_text = ""

        # gather headline information
        title = item.find("h3")
        if title is None:
            count_errors += 1
            error_text += "h3 not found. "
            continue
        else:
            title = title.contents[0]
        if title in df_company.values:
            continue
        date_str = item.find("span", attrs={"class": "date"}).contents[0].strip()
        published_on = datetime.strptime(date_str, '%d.%m.%Y').date()
        article_url = item['href']
        article_category = item.find("div", attrs={"class": "target-group"}).contents[0]

        # go to article page
        page = s.get(article_url).content
        sub_soup = BeautifulSoup(page, 'html.parser')

        # get pr pdf
        pdf = sub_soup.find("a", attrs={"aria-label": "Press release"})
        successful = False
        if pdf is not None:
            try:
                pdf_url = base_link + pdf['href']
                successful = True
            except:
                pdf_url = None

        if not successful:
            pdf_url = None
            error_text += "pdf_url not found. "
            count_errors = count_errors + 1

        # get article text
        try:
            relevant_part_str = str(sub_soup).split('<h2 class="headline">')[1].split('<div data-tpl="con01">')[0]
        except:
            relevant_part_str = str(sub_soup)
            error_text += "Relevant part not determined. "
            print("RWE: Error with getting relevant part")
        article_text = h.handle(relevant_part_str).replace("\n", " ").strip()

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'article_url', 'article_category',
                                               'article_text', 'pdf_url', 'error_text', 'language']] =\
            [published_on, title, article_url, article_category, article_text, pdf_url, error_text, "English"]

        if len(df_company.index) >= max_articles:
            print("Gathered RWE headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered RWE headlines with " + str(count_errors) + " error(s) and " + str(count_warnings) + " warning(s).")
    return df_company


def getEphNews(soup, page):
    return None


def getPgeNews(soup, page):
    return None


def getArcelorNews(soup, page, max_articles=1000):
    company = "Arcelor"
    base_url = "https://corporate.arcelormittal.com"
    language = "English"
    date_format = "%Y-%m-%d"
    article_tag = "article"
    article_class = "card grid-2x2__region"
    time_tag = "time"
    relevant_article_start = '<div class="content-main">'
    relevant_article_end = '<div class="article-disclaimer">'
    relevant_article_end2 = 'View all Press Releases'
    relevant_article_end3 = '<div class="article-aside">'

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.body_width = 0

    s = requests.Session()
    #s.headers.update({'User-Agent': 'Mozilla/5.0'})
    #s.cookies.update({'gdpr-settings-cookie': '1|true|16.8.2022|16.8.2024|1059955080'})

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all(article_tag, attrs={"class": article_class})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 20 == 0:
            print(company + " article " + str(len(df_company.index)))
        error_text = ""

        # gather headline information
        title_elem = item.find("h3").find("a")
        if title_elem is None:
            count_errors += 1
            error_text += "title not found. "
            continue
        else:
            title = title_elem.contents[0]
        if title in df_company.values:
            continue
        date_str = item.find(time_tag)["datetime"]
        published_on = datetime.strptime(date_str, date_format).date()
        article_url = base_url + title_elem["href"]

        # go to article page
        page = s.get(article_url).content
        sub_soup = BeautifulSoup(page, 'html.parser')

        # get summary text
        preview_div = sub_soup.find("div", attrs={"class": "article-standfirst"})
        if preview_div is not None:
            preview_text = preview_div.get_text().strip()
        else:
            preview_text = None

        # get article text
        try:
            relevant_part_str = str(sub_soup).split(relevant_article_start)[1]
            if relevant_part_str.find(relevant_article_end) != -1:
                relevant_part_str = relevant_part_str.split(relevant_article_end)[0]
            if relevant_part_str.find(relevant_article_end2) != -1:
                relevant_part_str = relevant_part_str.split(relevant_article_end2)[0]
            if relevant_part_str.find(relevant_article_end3) != -1:
                relevant_part_str = relevant_part_str.split(relevant_article_end3)[0]
        except:
            relevant_part_str = str(sub_soup)
            error_text += "Relevant part not determined. "
            print(company + ": Error with getting relevant part")
        #for script in soup(["script", "style"]):
        #    script.extract()  # delete
        article_text = h.handle(relevant_part_str)
        article_text = article_text.replace("\n", " ").strip()

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'preview_text', 'article_url',
                                               'article_text', 'error_text', 'language']] = \
            [published_on, title, preview_text, article_url, article_text, error_text, language]

        if len(df_company.index) >= max_articles:
            print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(count_warnings) + " warning(s).")
    return df_company


def getEngieNews(soup, page):
    return None


def getHeidelbergNews(soup, page, max_articles=1000):
    rank, name, category, language, sector, url, html, scrape = page

    company = "HeidelbergCem."
    base_url = "https://www.heidelbergcement.com"
    language = "English"
    article_tag = "div"
    article_class = "hc-teaser__content"

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.body_width = 0

    s = requests.Session()

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all(article_tag, attrs={"class": article_class})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 5 == 0:
            print(company + " article " + str(len(df_company.index)) + " of " + str(len(news)))
        error_text = ""

        # gather headline information
        title_elem = item.find("a", attrs={"class": "hc-link"})
        article_url = base_url + title_elem['href']
        title = title_elem.contents[1].contents[0]
        date_str = item.find("ul").find_all("li")[1].contents[0].strip()
        published_on = datetime.strptime(date_str, '%d %B %Y').date()

        # go to article page
        page = s.get(article_url).content
        sub_soup = BeautifulSoup(page, 'html.parser').find("div", attrs={"class": "hc-bodytext"})
        #unwanted = sub_soup.find_all("sup")
        #for elem in unwanted:
        #    elem.extract()
        preview_text = None

        # get article text
        article_text = h.handle(str(sub_soup))
        article_text = article_text.replace("\n", " ").strip()

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'preview_text', 'article_url',
                                               'article_category', 'article_text', 'error_text', 'language']] = \
            [published_on, title, preview_text, article_url, category, article_text, error_text, language]

        if len(df_company.index) >= max_articles:
            print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
        count_warnings) + " warning(s).")
    return df_company


def getTotalNews(soup, page, max_articles=100):
    rank, name, category, language, sector, url, html, scrape = page

    company = "Total"
    base_url = "https://totalenergies.com"
    language = "English"
    article_tag = "article"
    article_class = "press_release"

    #scraper = cs.CloudScraper(browser={'browser': 'firefox', 'platform': 'windows', 'mobile': False})  # CloudScraper inherits from requests.Session
    #scraper.headers.update({'User-Agent': 'Mozilla/5.0'})

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.body_width = 0
    h.ignore_images = True
    h.ignore_tables = True

    client = httpx.Client(http2=True)

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all(article_tag, attrs={"class": article_class})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 5 == 0:
            print(company + " article " + str(len(df_company.index)) + " of " + str(len(news)))
        error_text = ""

        # gather headline information
        title_elem = item.find("h2")
        title = title_elem.get_text().strip()
        article_url = base_url + title_elem.find("a")['href']

        # go to article page
        #page = scraper.get(article_url).text
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}
        page = client.get(article_url, headers=headers)
        if str(page) == "<Response [429 Too Many Requests]>":
            print("Too many requests")
            return df_company
        sub_soup = BeautifulSoup(page, 'html.parser').find("article")
        if sub_soup is None:
            count_errors += 1
            print(company + ": Error with getting article " + article_url)
            print(page)
            print(sub_soup)
            continue

        # get pdf link
        pdf_url = base_url + sub_soup.find("a")["href"]

        # get article text
        article_text = h.handle(str(sub_soup)).replace("\n", " ").strip()
        try:
            article_text = article_text.split("About TotalEnergies")[0]
        except:
            None
        try:
            article_text = article_text.split("KB)")[1]
        except:
            None

        # get date
        date_str = sub_soup.find("div", attrs={"class": "article-intro--date"}).get_text()
        published_on = datetime.strptime(date_str, "%m/%d/%Y").date()

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'article_url',
                                               'article_category', 'article_text', 'error_text', 'language',
                                               'pdf_url']] = \
            [published_on, title, article_url, category, article_text, error_text, language, pdf_url]

        if len(df_company.index) >= max_articles:
            print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
        count_warnings) + " warning(s).")
    return df_company


def getHolcimNews(soup, page, max_articles=1000):
    rank, name, category, language, sector, url, html, scrape = page

    company = name
    language = "English"
    article_tag = "div"
    article_class = "title_wrapper"
    base_url = "https://www.holcim.com"

    s = requests.Session()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.ignore_tables = True
    h.ignore_images = True
    h.body_width = 0

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all(article_tag, attrs={"class": article_class})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 5 == 0:
            print(company + " article " + str(len(df_company.index)) + " of " + str(len(news)))
        error_text = ""

        # gather headline information
        title_elem = item.find("a")
        title = title_elem.get_text().strip()
        article_url = title_elem['href']

        # get date
        try:
            date_str = item.find("div", attrs={"class": "date"}).contents[0]
            published_on = datetime.strptime(date_str, "%d %B %Y").date()
        except:
            print(company + ": Error with date. ")
            count_errors += 1
            continue

        # go to article page
        page = s.get(article_url).content
        sub_soup = BeautifulSoup(page, 'html.parser')
        if sub_soup is None:
            count_errors += 1
            print(company + ": Error with getting article " + article_url)
            continue

        # get pdf link
        try:
            try:
                pdf_url = sub_soup.find("ul", attrs={"class": "cta_links"}).find("a")["href"]
            except:
                pdf_url = sub_soup.find("ul", attrs={"class": "download_files"}).find("a")["href"]
            if pdf_url[:4] != "http":
                pdf_url = base_url + pdf_url
        except:
            pdf_url = None
            error_text += "No pdf url. "

        # get article text and preview
        try:
            text_elem = sub_soup.find_all("div", attrs={"class": "text"})[1]
            if text_elem.get_text().strip() == "":
                text_elem = sub_soup.find_all("div", attrs={"class": "text"})[3]
        except:
            continue
        preview_elem = text_elem.find("ul")
        if preview_elem is not None:
            try:
                preview_list = ["".join(elem.find_all(text=True)) for elem in preview_elem.find_all("strong")]
                if len(preview_list) == 0:
                    raise Exception
                if preview_list[0][0].strip() == '(':
                    raise Exception
                preview_text = "- " + "\n- ".join(preview_list)
            except:
                preview_text = None
                error_text += "Error with getting preview text. "
                count_errors += 1
        else:
            preview_text = None
            error_text += "No preview text. "
        article_text = reduce(lambda a, b: a + " " + b if a[-1] == '.' or a[-2:] == '."' else a + ". " + b,
                              [x.strip() for x in
                               h.handle(str(text_elem)).replace('”', '"').replace('“', '"').strip().splitlines()
                               if x.strip() != ""])
        try:
            article_text = article_text.split("Important disclaimer – forward-looking statements:")[0]
        except:
            None
        if article_text == "":
            error_text += "Error with Article text. "
            count_errors += 1
            print(sub_soup.find_all("div", attrs={"class": "text"}))

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'article_url',
                                               'article_category', 'preview_text', 'article_text', 'error_text',
                                               'language', 'pdf_url']] = \
            [published_on, title, article_url, category, preview_text, article_text, error_text, language, pdf_url]

        if len(df_company.index) >= max_articles:
            print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
        count_warnings) + " warning(s).")
    return df_company


def getThyssenNews(soup, page, max_articles=1000):
    rank, name, category, language, sector, url, html, scrape = page

    language = "English"
    article_tag = "div"
    article_class = "press-teaser"
    base_url = "https://www.thyssenkrupp.com"

    s = requests.Session()

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all(article_tag, attrs={"data-cy": article_class})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 5 == 0:
            print(name + " article " + str(len(df_company.index)) + " of " + str(len(news)))
        error_text = ""

        # gather headline information
        title_list = item.find("div", attrs={"class": "caption"}).find_all("p")
        title = title_list[1].contents[0].strip()
        article_url = base_url + item.find("a")['href']
        category = "All"
        metadata = title_list[0].contents[0].split(" | ")
        if len(metadata) > 1:
            article_categories = metadata[1].split(", ")
            for cat in article_categories:
                if cat != "Company News":
                    category = cat

        # get date
        try:
            date_str = title_list[0].contents[0][:8]
            published_on = datetime.strptime(date_str, "%d.%m.%y").date()
        except:
            print(name + ": Error with date. ")
            print(article_url)
            count_errors += 1
            continue

        # go to article page
        page = s.get(article_url).content
        sub_soup = BeautifulSoup(page, 'html.parser')
        if sub_soup is None:
            count_errors += 1
            print(name + ": Error with getting article " + article_url)
            continue

        pdf_url = None

        # get article text and preview
        text_list = sub_soup.find_all("ucp-standard-text")

        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_emphasis = True
        h.body_width = 0

        article_text = "\n\n".join([h.handle(str(text_elem)).strip() for text_elem in text_list])
        article_text = article_text.replace("\n\n", "\n").replace("\n\n", "\n")
        unwanted = ["\"This document includes “'forward-looking statements", "https://www.thyssenkrupp.com",
                    "http://www.thyssenkrupp.com", "\nContact", "\nContact details:\n",
                    "\nGo here for current footage material.\n", "\nPictures can be found here.\n",
                    "\nMore information at", "www.thyssenkrupp", "\n______________", "Further information",
                    "More information at", "Link to press release", "Please find here press photos for download"]
        for elem in unwanted:
            try:
                cut_index = article_text.index(elem)
                article_text = article_text[:cut_index]
            except:
                cut_index = None
        article_text = article_text.strip()
        preview_text = None

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'article_url',
                                               'article_category', 'preview_text', 'article_text', 'error_text',
                                               'language', 'pdf_url']] = \
            [published_on, title, article_url, category, preview_text, article_text, error_text, language, pdf_url]

        if len(df_company.index) >= max_articles:
            print("Gathered " + name + " headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered " + name + " headlines with " + str(count_errors) + " error(s) and " + str(
        count_warnings) + " warning(s).")
    return df_company


def getOrlenList():
    s = requests.Session()
    archive_url = "https://a.orlen.pl/EN/PressOffice/Pages/PressReleases.aspx?dateFrom=01-01-2000&dateTo=25-08-2022&start="
    f = open("../PageHtml/15_orlen.txt", "a")

    for i in range(0, 1000, 5):
        print(i)
        page = s.get(archive_url + str(i)).content
        soup = BeautifulSoup(page, 'html.parser')
        list = soup.find_all("div", attrs={"class": "search_results_box"})
        out = "\n".join([str(elem) for elem in list])
        f.write(out + "\n\n")

    f.close()
    return None


def getOrlenNews(soup, page, max_articles=1000):
    rank, name, category, language, sector, url, html, scrape = page

    company = "Orlen"
    base_url = "https://a.orlen.pl"
    language = "English"
    article_tag = "div"
    article_class = "search_results_box"

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all(article_tag, attrs={"class": article_class})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 5 == 0:
            print(company + " article " + str(len(df_company.index)) + " of " + str(len(news)))
        error_text = ""

        # gather headline information
        links = item.find_all("a")
        title = links[0].contents[0]
        article_url = base_url + links[0]['href']
        try:
            preview_text = links[1].contents[0]
        except:
            error_text += "No preview text. "
            preview_text = None
        date_str = item.find("span").contents[0].strip()[:10]
        published_on = datetime.strptime(date_str, "%d-%m-%Y").date()

        # get article text
        sub_soup = BeautifulSoup(requests.get(article_url).text, 'html.parser')
        sub_soup = sub_soup.find("div", attrs={"content_site_news"})
        sub_soup.find("div", attrs={"class": "newsImage"}).extract()
        sub_soup.find("div", attrs={"style": "display:none"}).extract()
        sub_soup.find("a", attrs={"href": "default.aspx"}).extract()

        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_emphasis = True
        h.body_width = 0

        article_text = h.handle(str(sub_soup)).strip()
        pdf_url = None

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'article_url',
                                               'article_category', 'article_text', 'preview_text', 'error_text',
                                               'language', 'pdf_url']] = \
            [published_on, title, article_url, category, article_text, preview_text, error_text, language, pdf_url]

        if len(df_company.index) >= max_articles:
            print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered " + company + " headlines with " + str(count_errors) + " error(s) and " + str(
        count_warnings) + " warning(s).")
    return df_company


def getExxonNews(soup, page, max_articles=1000):
    rank, name, category, language, sector, url, html, scrape = page

    base_url = "https://corporate.exxonmobil.com"
    language = "English"
    article_tag = "div"
    article_class = "contentCollection--content"

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.body_width = 0

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all(article_tag, attrs={"class": article_class})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 5 == 0:
            print(name + " article " + str(len(df_company.index)) + " of " + str(len(news)))
        error_text = ""

        # gather headline information
        link_elem = item.find("a")
        title = link_elem.contents[0].strip()
        article_url = base_url + link_elem['href']
        try:
            preview_text = item.find("span").contents[0]
            if type(preview_text) != "str":
                preview_text = h.handle(str(preview_text))
        except:
            error_text += "No preview text. "
            preview_text = None
        date_str = item.find("span", attrs={"class": "date"}).contents[0].strip()
        try:
            published_on = datetime.strptime(date_str, "%B %d, %Y").date()
        except:
            try:
                published_on = datetime.strptime(date_str, "%b. %d, %Y").date()
            except:
                try:
                    published_on = datetime.strptime(date_str, "%bt. %d, %Y").date()
                except:
                    error_text += "Error with date. "
                    count_errors += 1
                    published_on = None
                    print(name + ": Error with date. ")
                    print(article_url)
                    print(date_str)

        # get article text
        sub_soup = BeautifulSoup(requests.get(article_url).text, 'html.parser')
        sub_soup = sub_soup.find("div", attrs={"class": "article--wrapper"})
        if sub_soup is None:
            print(name + ": Error with article text. ")
            print(article_url)
            continue
        sub_soup = "".join([str(elem) for elem in sub_soup.find_all("section", attrs={"class": "rich-text"})])

        article_text = h.handle(sub_soup)
        if preview_text is not None:
            article_text = preview_text + article_text
        article_text = article_text.replace("\n\n", "\n").replace("\n\n", "\n").replace("#", "")
        unwanted = ["Cautionary Statement", "About ExxonMobil"]
        for elem in unwanted:
            try:
                cut_index = article_text.index(elem)
                article_text = article_text[:cut_index]
            except:
                cut_index = None
        pdf_url = None

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'article_url',
                                               'article_category', 'article_text', 'preview_text', 'error_text',
                                               'language', 'pdf_url']] = \
            [published_on, title, article_url, category, article_text, preview_text, error_text, language, pdf_url]

        if len(df_company.index) >= max_articles:
            print("Gathered " + name + " headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered " + name + " headlines with " + str(count_errors) + " error(s) and " + str(
        count_warnings) + " warning(s).")
    return df_company


def getBasfNews(soup, page, max_articles=1000):
    rank, name, category, language, sector, url, html, scrape = page

    base_url = "https://www.basf.com"
    language = "English"
    article_tag = "div"
    article_class = "news-result"

    s = requests.Session()
    s.headers.update({'User-Agent': 'Mozilla/5.0'})

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_emphasis = True
    h.body_width = 0

    df_company \
        = pd.DataFrame(data=None, columns=['published_on', 'title', 'preview_text', 'article_text',
                                           'article_url', 'article_category', 'pdf_url', 'error_text', 'language'])
    news = soup.find_all(article_tag, attrs={"class": article_class})
    count_errors = 0
    count_warnings = 0
    for item in news:
        if len(df_company.index) % 5 == 0:
            print(name + " article " + str(len(df_company.index)) + " of " + str(len(news)))
        error_text = ""

        # gather headline information
        category = item.find("span", attrs={"class": "news-result__category"}).contents[0]
        title = item.find("h3").contents[0]
        article_url = base_url + item.find("a")['href']
        #preview_text = item.find("div", attrs={"class": "search-result__description"}).contents[0]
        date_str = item.find("span", attrs={"class": "news-result__date"}).contents[0]
        try:
            published_on = datetime.strptime(date_str, "%B %d, %Y").date()
        except:
            error_text += "Error with date. "
            count_errors += 1
            published_on = None
            print(name + ": Error with date. ")
            print(article_url)
            print(date_str)

        # get article text
        sub_soup = BeautifulSoup(s.get(article_url).text, 'html.parser')
        sub_soup = sub_soup.find("div", attrs={"class": "text component textbody"})
        if sub_soup is None:
            print(name + ": Error with article text. ")
            print(article_url)
            continue

        unwanted = sub_soup.find_all("span", attrs={"class": "cmp-text__footnote"})
        for elem in unwanted:
            elem.extract()

        ul_elem = sub_soup.find("ul")
        preview_text = h.handle(str(ul_elem))

        article_text = h.handle(str(sub_soup).replace('”', '"').replace('“', '"'))
        #if preview_text is not None:
        #    article_text = preview_text + article_text
        article_text = article_text.replace("\n\n", "\n").replace("\n\n", "\n").replace("#", "")
        unwanted = ["Press contact", "BASF Media Contact", "Media Relations", "---|---"]
        for elem in unwanted:
            try:
                cut_index = article_text.index(elem)
                article_text = article_text[:cut_index]
            except:
                cut_index = None

        pdf_url = None
        possible_pdf = sub_soup.find_all("div", attrs={"class": "teaser__link"})
        for elem in possible_pdf:
            link = elem["href"]
            if link.endswith(".pdf"):
                pdf_url = link

        # insert into df
        df_company.loc[len(df_company.index), ['published_on', 'title', 'article_url',
                                               'article_category', 'article_text', 'preview_text', 'error_text',
                                               'language', 'pdf_url']] = \
            [published_on, title, article_url, category, article_text, preview_text, error_text, language, pdf_url]

        if len(df_company.index) >= max_articles:
            print("Gathered " + name + " headlines with " + str(count_errors) + " error(s) and " + str(
                count_warnings) + " warning(s).")
            return df_company

    print("Gathered " + name + " headlines with " + str(count_errors) + " error(s) and " + str(
        count_warnings) + " warning(s).")
    return df_company


def getAllNewsDf(insert=True):
    noadb = dbm.get_database(Database.NOADB)
    td = getTableNames()
    fd = getFunctionDict()
    pages_in_tbl = td['pagetracker_nk_pages_in']
    pages_out_tbl = td['pagetracker_nk_out']

    sql = f'''SELECT * FROM {pages_in_tbl} WHERE scrape = 1'''
    pages_in_df = noadb.get_as_df(sql)

    df_all = pd.DataFrame(data=None, columns=['emissions_rank', 'company_name', 'article_category', 'published_on',
                                              'title', 'preview_text', 'article_text', 'article_url',
                                              'pdf_url', 'language'])

    for page in pages_in_df.values:
        rank, name, category, language, sector, url, html, scrape = page
        if html is None:
            soup = None
        else:
            soup = BeautifulSoup(html, 'html.parser')
        if rank == 1 and language == "English":
            df_company = getRweNewsEnglish(soup, page)
        else:
            df_company = fd[rank](soup, page)
        df_company.loc[:, ['emissions_rank', 'company_name']] = [rank, name]
        df_all = pd.concat([df_all, df_company])

    if insert:
        noadb.insert_df(pages_out_tbl, df_all)
    return df_all


if __name__ == '__main__':
    print(getAllNewsDf(True).to_string())