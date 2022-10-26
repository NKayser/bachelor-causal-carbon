"""
author: lischkers
last updated: 2022-02-17 by merschelp --> added ENVI Committee
"""

import numpy as np
import requests
from bs4 import BeautifulSoup
from GeneralInfrastructure.RunTimeController import RunTimeController
from GeneralInfrastructure import CustomLogger
from GeneralInfrastructure.CustomThread import CustomThread
from datetime import datetime, timedelta
import pandas as pd
import ast
import GeneralNoaDB.DBTableNames as tb
from GeneralInfrastructure.Database.DatabaseManager import dbm, Database
from Carbon.HtmlTableGenerator.generateHtmlTables import createHtmlTable
import logging


def runTracking():
    """
    This function scrapes all information from the websites specified in the database table and sends an update to
    the listed email recipients if new information has been published.
    """
    # ________________________________________________Initialization____________________________________________________
    pd.set_option('display.max_colwidth', -1)
    logger = logging.getLogger(__name__)
    noa_db = dbm.get_database(Database.NOADB)

    html_property_dic = {'darkGreyRows': [0],
                         'lightGreyRows': [],
                         'rightBlueBorderColumns': [0],
                         'boldColumns': [], 'emptyRows': [],
                         'mergedRows': [],
                         'mergedColumns': [],
                         'darkGreyLowerRows': [],
                         'indentRows': []}
    column_width_arr = [10, 25, 5]

    mail_subject = 'News from the page tracker'
    company_df = getCompaniesAndLinks()
    company_df.index = company_df['company']
    company_dic = company_df[['sector', 'link', 'emailReceiver']].to_dict(orient='index')
    email_text = 'Hi! <br><br>New information has been published on a webpage that you have signed up to:<br><br>'
    error_text = ''
    df_all = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])

    # __________________________________________________Web scraping____________________________________________________
    for company in company_dic:
        logger.info('Tracking company ' + company)
        sector = company_dic[company]['sector']
        url = company_dic[company]['link']
        email_receiver = company_dic[company]['emailReceiver']
        r1 = requests.get(url)
        cover_page = r1.content
        soup = BeautifulSoup(cover_page, 'html.parser')
        if company == 'ArcelorMittal':
            df_company, error_text = getArcelorMittalNews(soup, company, url, error_text)

        elif company == 'ICE':
            df_company, error_text = getIceNews(company, url, error_text)

        elif company == 'EC allocation':
            df_company, error_text = getECallocationNews(soup, company, url, error_text)

        elif company == 'EC Climate Action':
            df_company, error_text = getECclimateActionNews(soup, company, url, error_text)

        elif company == 'HeidelbergCement':
            df_company, error_text = getHeidelbergCementNews(soup, company, url, error_text)

        elif company == 'GOV_UK':
            df_company, error_text = getGovUKNews(soup, company, url, error_text)

        elif company == 'ENVI Committee':
            df_company, error_text = getENVINews(soup, company, url, error_text)

        elif company in ['EC MSR', 'EC Phase 4', 'EC Internal Credits', 'EC Auctioning', 'EC Free Allocation',
                         'EC Monitoring', 'EC Registry', 'EC Integrity', 'EC International', 'EC Development']:
            df_company, error_text = getECsubpageNews(soup, company, url, error_text)

        df_company.loc[:, ['company', 'sector', 'emailReceiver']] = [company, sector, email_receiver]
        df_all = pd.concat([df_all, df_company])

    df_all['publishedOn'] = pd.to_datetime(df_all['publishedOn'])
    previous_tracked_news_df = getTrackedNews()
    all_news_df = previous_tracked_news_df.merge(df_all, on=['company', 'title', 'publishedOn'], how='outer',
                                                 indicator=True)
    all_news_df = all_news_df.drop(columns=['link_x'])
    all_news_df = all_news_df.rename(columns={'link_y': 'link'})
    new_news_df = all_news_df.loc[(all_news_df.publishedOn > datetime.today() - timedelta(days=10))
                                  & (all_news_df._merge == 'right_only'), :]
    all_links_df = previous_tracked_news_df.merge(df_all, on=['company', 'title', 'link', 'publishedOn'], how='outer',
                                              indicator=True)
    updated_links_df = all_links_df.loc[(all_links_df.publishedOn > datetime.today() - timedelta(days=10))
                                    & (all_links_df._merge == 'right_only'), :]
    # only consider links as 'updated' if they are not listed in the new_news_df
    updated_links_df = updated_links_df.drop(columns=['_merge'])
    new_news_df = new_news_df.drop(columns=['_merge'])
    updated_links_df = updated_links_df.merge(new_news_df,
                                              on=['company', 'title', 'link', 'publishedOn', 'sector', 'emailReceiver'],
                                              how='outer', indicator=True)
    updated_links_df = updated_links_df.loc[updated_links_df._merge == 'left_only', :]

    # _____________________________Create HTML table with most recent information per sector____________________________
    all_news_df = all_news_df.merge(company_df[['sector']], on='company')
    all_news_df['sector'] = all_news_df['sector_y']
    all_news_df['titleLink'] = all_news_df.apply(lambda x: make_clickable(x['link'], x['title']), axis=1)
    all_news_df['publishedOn'] = pd.to_datetime(all_news_df['publishedOn'], format='%Y-%b-%d')
    for sector in all_news_df.sector.unique():
        logger.info('Creating Html tables for ' + sector)
        all_news_sector_df \
            = all_news_df.loc[(all_news_df.sector == sector),['company', 'titleLink', 'publishedOn']].sort_values(
            'publishedOn', ascending=False).head(7)
        all_news_sector_df['publishedOn'] = all_news_sector_df.publishedOn.astype(str)
        all_news_sector_array = [['company', 'link', 'published on']] + all_news_sector_df.values.tolist()
        html_str = '<h3>' + sector + '</h3>' \
                   + createHtmlTable(html_property_dic, all_news_sector_array, column_width_arr, False)
        out_df = pd.DataFrame(data=[[datetime.today(), sector, html_str]],
                              columns=['modelrun', 'sector', 'htmlTable'])

        # _______________________________________Save HTML table to database____________________________________________
        pagetracker_trackedNewsHtml_web = tb.pagetracker_trackednewshtml_web(tb.UNSYNCED)
        sql = 'SELECT * FROM {table} LIMIT 1'.format(table=pagetracker_trackedNewsHtml_web)
        field_list = noa_db.get_as_tuples(sql)[1]
        out_df = out_df.reindex(columns=field_list)
        noa_db.insert_df(pagetracker_trackedNewsHtml_web, out_df)

    # ___________________________________________New information --> send email_________________________________________
    if not new_news_df.empty:
        logger.info('Writing email')
        new_news_df['titleLink'] = new_news_df.apply(lambda x: make_clickable(x['link'], x['title']), axis=1)
        for sector in new_news_df.sector.unique():
            new_news_sector_df \
                = new_news_df.loc[(new_news_df.sector == sector), ['company', 'titleLink', 'publishedOn']].sort_values(
                'publishedOn', ascending=False)
            email_text = email_text + '<br>Sector: ' + sector + '<br>' \
                         + new_news_sector_df.to_html(index=False, render_links=True, escape=False)

        mail_text_final = email_text + '<br><br><br>_ _ _<br>' + error_text
        email_receivers = [[name.replace(' ', '.') + '@icis.com' for name in receiverList.split(', ')] for receiverList
                           in list(new_news_df.emailReceiver.unique())][0]
        logger.email(email_receivers, mail_subject, mail_text_final, msgType='html')

        new_news_df['retrievedOn'] = datetime.today()

        # _______________________________________Save new information to database_______________________________________
        pagetracker_trackednews_out = tb.pagetracker_trackednews_out(tb.UNSYNCED)
        sql = 'SELECT * FROM {table} LIMIT 1'.format(table=pagetracker_trackednews_out)
        field_list = noa_db.get_as_tuples(sql)[1]
        new_news_df = new_news_df.reindex(columns=field_list)
        logger.debug(new_news_df.to_string())
        noa_db.insert_df(pagetracker_trackednews_out, new_news_df)

    # ______________________________________________Error --> send email________________________________________________
    elif error_text != '':
        logger.debug(error_text)
        email_receivers = ['haeyun.park@icis.com', 'patricia.merschel@icis.com']
        logger.email(email_receivers, 'Page tracker errors', error_text, msgType='html')

    # __________________________________________Save updated links to database__________________________________________
    if not updated_links_df.empty:
        logger.info('Saving updated links to the database')
        pagetracker_trackednews_out = tb.pagetracker_trackednews_out(tb.UNSYNCED)
        updated_links_df['retrievedOn'] = datetime.now().date()
        updated_links_df['comment'] = np.nan
        noa_db.insert_df(pagetracker_trackednews_out,
                         updated_links_df[['company', 'title', 'link', 'publishedOn', 'retrievedOn', 'comment']])

    logger.info('DONE :-)')


def make_clickable(url, name):
    # add html code to make a weblink clickable
    return '<a href="{}">{}</a>'.format(url, name)


def getArcelorMittalNews(soup, company, url, error_text):
    # web scraper for updates on Arcelor Mittal website
    try:
        df_company \
            = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])
        base_link = 'https://corporate.arcelormittal.com'
        text = soup.find_all('article-latest')[0]
        dic = ast.literal_eval(text.attrs[':data'])['results']['items']
        for postNo in range(len(dic)):
            link = base_link + dic[postNo]['link']['href']
            title = dic[postNo]['heading']
            date_published = dic[postNo]['datetime']['iso']
            error_text = checkIfValidDate(date_published, company, url, error_text)
            df_company.loc[len(df_company.index), ['title', 'link', 'publishedOn']] = [title, link, date_published]

        error_text = checkEmptyDataFrame(df_company, error_text, company, url)

    except:
        error_text = error_text + '<br>' + 'Error for ' + company + '. Check website ' + url

    return df_company, error_text


def getECallocationNews(soup, company, url, error_text):
    # web scraper for updates on EC allocation website
    try:
        df_company \
            = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])
        relevant_part = soup.prettify().split('Documentation')[1]
        soup_new = BeautifulSoup(relevant_part, 'html.parser')
        news_list = soup_new.find('ul').find_all('li')
        for i in range(len(news_list)):
            content = news_list[i].contents
            date_published = content[0].split('-')[0].replace(' ', '').replace('\n', '').replace('-', '')
            try:
                date_published = pd.to_datetime(date_published, format='%d/%m/%Y')
            except:
                pass
            error_text = checkIfValidDate(date_published, company, url, error_text)
            link = content[1]['href']
            title = content[1].text.replace('  ', '').replace('\n', '').replace('-', '')
            df_company.loc[len(df_company.index), ['title', 'link', 'publishedOn']] = [title, link, date_published]

        error_text = checkEmptyDataFrame(df_company, error_text, company, url)

    except:
        error_text = error_text + '<br>' + 'Error for ' + company + '. Check website ' + url

    return df_company, error_text


def getECclimateActionNews(soup, company, url, error_text):
    # web scraper for updates on EC climate action website
    try:
        df_company \
            = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])

        base_link = 'https://ec.europa.eu'

        for item in soup.find_all('div', {'class': 'ecl-u-flex-grow-1'}):
            date_published = item.find_all('time')[0].text
            title = item.find_all('a')[0].text
            link = item.find_all('a')[0].get('href')
            if link[0:8] != 'https://':
                link = base_link + link

            df_company.loc[len(df_company.index), ['title', 'link', 'publishedOn']] = [title, link, date_published]

        error_text = checkEmptyDataFrame(df_company, error_text, company, url)

    except:
        error_text = error_text + '<br>' + 'Error for ' + company + '. Check website ' + url

    return df_company, error_text


def getHeidelbergCementNews(soup, company, url, error_text):
    # web scraper for updates on Heidelberg Cement website
    try:
        df_company \
            = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])
        news = soup.find_all("div", attrs={"class": "hc-teaser__content"})
        base_link = 'https://www.heidelbergcement.com'
        for item in news:
            pr = item.find_all('a', {'class': 'hc-link'})[0]
            link = base_link + pr['href']
            title = pr.contents[1].contents[0]
            date_str = pr['href'].replace('/en/pr-', '').replace('/en/pi-', '')
            date_published = datetime.strptime(date_str, '%d-%m-%Y').date()

            df_company.loc[len(df_company.index), ['title', 'link', 'publishedOn']] = [title, link, date_published]

        error_text = checkEmptyDataFrame(df_company, error_text, company, url)

    except:
        error_text = error_text + '<br>' + 'Error for ' + company + '. Check website ' + url

    return df_company, error_text


def getIceNews(company, companyLink, error_text):
    # web scraper for updates on Ice website
    import json
    try:
        df_company \
            = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])
        url = 'https://www.theice.com/notices/FuturesEuropeCirculars.shtml'
        requestHeader = {'Host': 'www.theice.com',
                         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:76.0) Gecko/20100101 Firefox/76.0',
                         'Accept': '*/*',
                         'Accept-Language': 'de,en-US;q=0.7,en;q=0.3',
                         'Accept-Encoding': 'gzip, deflate, br',
                         'Referer': 'https://www.theice.com/futures-europe/circulars',
                         'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                         'X-Requested-With': 'XMLHttpRequest'}
        payload = {'criteria.year': datetime.today().date().year,
                   'criteria.topic': 'Auctions',
                   'noticeDataAsHTML': '',
                   '_sourcePage': 'PzDOP2k2xs5lFkwyeOG-h8DE8k9zlZhDX922h5FVILFrU0g1PbIvRKCkLuXMQh-Uc8Fttis5pd9hPzxr9PIYbBSpJdUxygd-',
                   '__fp': 'wxUAW4d40b-pgzpFPi3Vs0-Yzxs5m4JviD3oYj22e8swDa2V6CmtXA%3D%3D'}

        r = requests.post(url, data=json.dumps(requestHeader), params=payload)
        soup = BeautifulSoup(r.text, 'html.parser')
        news_list = soup.find_all('td', {'class': 'table-align-left'})
        base_link = 'https://www.theice.com'
        for i in range(0, len(news_list), 3):
            date_published = news_list[i].text.replace('  ', '').replace('\n', '')
            link = base_link + news_list[i+2].find_all('a')[0]['href']
            title = news_list[i+2].find_all('a')[0].text

            df_company.loc[len(df_company.index), ['title', 'link', 'publishedOn']] = [title, link, date_published]

        # dropdown menus are used --> empty df does not mean that scraper stopped working
#        error_text = checkEmptyDataFrame(df_company, error_text, company, url)

    except:
        error_text = error_text + '<br>' + 'Error for ' + company + '. Check website ' + companyLink

    return df_company, error_text


def getGovUKNews(soup, company, url, error_text):
    # web scraper for updates on GOV UK website
    try:
        df_company \
            = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])

        relevant_part = soup.findAll('div', {'class': 'app-c-published-dates__change-history', 'id': 'full-history'})
        news_items = relevant_part[0].findAll('ol')[0].findAll('li')

        link = 'https://www.gov.uk/government/publications/participating-in-the-uk-ets#full-publication-update-history'

        for item in news_items:
            date_published = item.find('time', {'class': 'app-c-published-dates__change-date timestamp'}).text
            title = item.find('p', {'class': 'app-c-published-dates__change-note'}).text

            df_company.loc[len(df_company.index), ['title', 'link', 'publishedOn']] = [title, link, date_published]

        error_text = checkEmptyDataFrame(df_company, error_text, company, url)

    except:
        error_text = error_text + '<br>' + 'Error for ' + company + '. Check website ' + url

    return df_company, error_text


def getENVINews(soup, company, url, error_text):
    # web scraper for updates on ENVI committee website
    try:
        df_company \
            = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])

        relevant_part = soup.find('main', {'id': 'website-body'})
        news_items = relevant_part.findAll('div', {'class': 'erpl_document'})

        for item in news_items:
            date_published = pd.to_datetime(item.find('span', {'class': 'erpl_document-subtitle-fragment'}).text,
                                            dayfirst=True)
            title = item.find('span', {'class': 't-item'}).text
            committee_badges = item.find_all('span', {'class': 'erpl_document-subtitle-badges'})
            if committee_badges:
                badges = committee_badges[0].find_all('a')
                for badge in badges:
                    title = title + '_' + badge.text.replace('\n', '').replace('\t', '')
            link = item.find('a', {'class': 'erpl_document-subtitle-pdf'}).get('href')

            df_company.loc[len(df_company.index), ['title', 'link', 'publishedOn']] = [title, link, date_published]

        error_text = checkEmptyDataFrame(df_company, error_text, company, url)

    except:
        error_text = error_text + '<br>' + 'Error for ' + company + '. Check website ' + url

    return df_company, error_text


def getECsubpageNews(soup, company, url, error_text):
    # web scraper for updates on EC allocation website
    try:
        df_company \
            = pd.DataFrame(data=None, columns=['company', 'sector', 'title', 'link', 'publishedOn', 'emailReceiver'])
        relevant_part = soup.prettify().split('Documentation')[1].split('block-ewcms-theme-socialshare')[0]
        soup_new = BeautifulSoup(relevant_part, 'html.parser')
        uls = soup_new.find_all('ul')
        news_list = []
        for ul in uls:
            news_list += (ul.find_all('li'))
        for i in range(len(news_list)):
            content = news_list[i].contents
            date_published = content[0].split('–')[0].split('-')[0].replace(' ', '').replace('\n', '')
            if date_published == '':
                continue
            try:
                if date_published.count('/') == 2:
                    date_published = pd.to_datetime(date_published, format='%d/%m/%Y')
                elif date_published.count('/') == 1:
                    date_published = pd.to_datetime(date_published, format='%m/%Y')
            except:
                continue
            if checkIfValidDate(date_published, company, url, "") != "":
                continue
            try:
                link = content[1]['href']
            except:
                continue
            if link[0] == '/':
                link = "https://ec.europa.eu" + link
            title = content[1].text.replace('  ', '').replace('\n', '').replace('-', '')
            df_company.loc[len(df_company.index), ['title', 'link', 'publishedOn']] = [title, link, date_published]

        error_text = checkEmptyDataFrame(df_company, error_text, company, url)

    except:
        error_text = error_text + '<br>' + 'Error for ' + company + '. Check website ' + url

    return df_company, error_text


def checkIfValidDate(date, company, url, email_text):
    # send an error message if a date format is not valid
    try:
        pd.to_datetime(date)
    except:
        email_text = email_text + ' check published on for ' + company + ' link: ' + url

    return email_text


def checkEmptyDataFrame(df_company, error_text, company, url):
    # if the dataframe is empty, the website was probably changed --> send error message to specified email receivers
    if df_company.empty:
        error_text = error_text + '<br>Page tracker cannot find data for ' + company + '.' + '<br>' \
                    + 'See if the specified website is still the current version and if any changes have been made. ' \
                      '<br>Link to website: ' + url
    else:
        error_text = ''

    return error_text


def getCompaniesAndLinks():
    # read the companies and the web address that should be tracked from the database
    noa_db = dbm.get_database(Database.NOADB)
    sql = '''
    SELECT company, sector, newsUrl as 'link', emailReceiver
    FROM %(pagetracker_pages_in)s
    '''%{'pagetracker_pages_in': tb.pagetracker_pages_in(tb.UNSYNCED)}
    df = noa_db.get_as_df(sql)

    return df


def getTrackedNews():
    noa_db = dbm.get_database(Database.NOADB)
    sql = '''
    SELECT company, title, link, publishedOn
    FROM %(pagetracker_trackednews_out)s
    ''' %{'pagetracker_trackednews_out': tb.pagetracker_trackednews_out(tb.UNSYNCED)}
    df = noa_db.get_as_df(sql)
    df['publishedOn'] = pd.to_datetime(df['publishedOn'])

    return df


def start():
    debugMode = True
    displayLogsOnConsole = True
    insertLogsIntoDatabase = False

    runTimeController1 = RunTimeController('mod_carbon_pagetracker', debugMode, insertLogsIntoDatabase, displayLogsOnConsole)
    errorQueue = None
    t1 = CustomThread(runTimeController=runTimeController1, errorQueue=errorQueue, target=runTracking, args=())
    t1.start()
    t1.join()
    # exc = t1.errorQueue.get()
    t1.raiseException()


if __name__ == '__main__':
    CustomLogger.initialize()

    start()
