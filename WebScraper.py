from bs4 import BeautifulSoup,SoupStrainer
import requests
import time
import csv
import os
from os import listdir
from os.path import isfile, join
import re

directoryPath_output = r"DIRECTORY WHERE YOU WANT TO WRITE SCRAPED TEXT TO"
fileNames  = [f for f in listdir(directoryPath_output) if isfile(join(directoryPath_output, f))]

'''Scan files in the output directory and compile a list of file names.
This list will be referenced to check if the article has already been scrapped and saved.
The article will not be scrapped if it already exists in the directory'''

filesInDirectory = []
for fileName in fileNames:
	split_fileName = fileName.split(".")
	filesInDirectory.append(split_fileName[0])
	
#scrollCount tells the script how many times to send a request to scroll to the next page in an infinite scroll website
scrollCount = 100
url_base = "WEBSITE URL"
page = 0
maxPage = int(page+scrollCount)

while page<=maxPage:
	page += 1
	url = ("%s%s%s%s%s" % (url_base,"page","/",str(page),"/"))
	try:
		response = requests.get(url)
		content = requests.get(url)
		soup = BeautifulSoup(content.text,'html.parser')
		articles_metaData = soup.findAll("p", {"class","card-article__description"})
		for article_metaData in articles_metaData:
			article_href = str((article_metaData.find(href=True))['href'])
			article_title = str(article_metaData.text)
			article_title = article_href.split("/")[-2]
			response_article = requests.get(article_href)
			content_article = requests.get(article_href)
			soup_article = BeautifulSoup(content_article.text,'html.parser')
			article_h1 = soup_article.find("h1", {"class","block-single__title"}).text
			article_p_1 = soup_article.find("p", {"class","block-single__excerpt"}).text
			article_topic = str(soup_article.find("div", {"class","block-single__topic"}).text).split(" ")[0].strip()
			article_date = soup_article.find("span", {"class","block-single__date"}).text
			article_content = soup_article.find("div", {"class","block-single__content"})
			if str("%s%s%s" % (article_topic,"_",article_title)) not in filesInDirectory:
				print(article_topic,article_title)
				with open("%s%s%s%s%s" % (directoryPath_output,article_topic,"_",article_title,".txt"), 'w', encoding='utf-8') as write_content:
					#wrte the article metadata to the txt file
					write_content.write("%s%s" % (article_href,"\n"))
					write_content.write("%s%s" % (article_h1,"\n"))
					write_content.write("%s%s" % (article_p_1,"\n"))
					write_content.write("%s%s" % (article_topic,"\n"))
					write_content.write("%s%s" % (article_date,"\n"))
					
					#write a tag denoting the begining of the article text to the txt file
					#this tag will be used in the text mining script to identify where the metadata ends and article text begins
					
					write_content.write("%s%s" % ("<TEXT BEGIN>","\n"))
					content_li = article_content.find_all("li")
					content_p = article_content.find_all("p")
					content_h2 = article_content.find_all("h2")
					for h2 in content_h2:
						write_content.write("%s%s" % (h2.text,"\n"))
					for li in content_li:
						write_content.write("%s%s" % (li.text,"\n"))
					store_pa_href = []
					store_pa_txt = []
					for p in content_p:
						write_content.write("%s%s" % (p.text,"\n"))
						p_a = p.find("a")
						if isinstance(p_a, type(None)) == False:
							store_pa_href.append(p.find("a")["href"])
							store_pa_txt.append(p.find("a").text)
							
					#write tags denoting the end of article text and the begining of link text
					
					write_content.write("%s%s" % ("<TEXT END>","\n"))
					write_content.write("%s%s" % ("<LINKS BEGIN>","\n"))
					for i in range(len(store_pa_href)):
						write_content.write("%s%s%s%s" % (str(store_pa_href[i]),",",str(store_pa_txt[i]),"\n"))
					write_content.write("%s%s" % ("<LINKS END>","\n"))
				write_content.close()
				time.sleep(1)
			get_infiniteScroll = soup.findAll("p", {"class", "infinitescroll"})
			#tell the script to sleep so you won't be blocked by the site for sending too many requests
			time.sleep(1)
	except Exception as e:
		print(e)
		continue
