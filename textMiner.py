import csv
import json
import math
import os
import re
import string
from datetime import datetime
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')
porterStemmer = PorterStemmer()
stemmer = SnowballStemmer("english", ignore_stopwords=True)

#set input/output directories
directoryPath_corpus = r"PATH TO DIRECTORY CONTAINING DOCUMENT CORPUS"
directoryPath_ledgers = r"PATH TO LEDGERS"
directoryPath_resource = r"PATH TO RESOURCES"

def check_encoding(file2check):
	file_encoding = ""
	with open(file2check) as file_info:
		file_encoding = file_info.encoding
	file_info.close()
	print(file_encoding)
	return str(file_encoding)

#read csv containing stopwords
filePath_resource_nltk = str("%s%s" % (directoryPath_resource,"stopwords.csv"))
with open(filePath_resource_nltk, 'r',encoding=check_encoding(filePath_resource_nltk)) as read_stopwords:
	reader = csv.reader(read_stopwords, delimiter=',')
	header = next(reader)
	stopwords = np.array(list(reader))
stopwordList = list(stopwords[:,0])

#read json containing terms, names and phrases to search for
filePath_text2search = str("%s%s" % (directoryPath_resource,"AEC Reference.json"))
with open(filePath_keywords2search, "r",encoding="utf8") as read_text2search:
	jsonData = read_text2search.read()
	text2search = json.loads(jsonData)
	
filePath_corpusLedger = str("%s%s" % (directoryPath_ledgers,"Corpus_Ledger_Revised.csv"))
print(filePath_corpusLedger)
with open(filePath_corpusLedger, 'r',encoding=check_encoding(filePath_corpusLedger)) as read_corpusLedger:
	reader = csv.reader(read_corpusLedger, delimiter=',')
	header = next(reader)
	corpusLedger = np.array(list(reader))

#"docId","doc_pupDate","docURL","docTopic","docTitle"
cl_docIds = corpusLedger[:,0]
cl_docIds_short = list(corpusLedger[:,1])
cl_docPupDate = corpusLedger[:,2]
cl_docTopic = corpusLedger[:,3]
cl_docTitle = corpusLedger[:,4]
cl_docUrl = corpusLedger[:,5]

def parseSearchText(input_raw):
	"""this function parses text from the json of text2search.
	Input text is split from strings into lists of individual words.
	Each individual word is appended to 'output_parsedSingles'
	and a correspinding list Id value is appended to 'output_refIds'
	and the list of individual words is appended to 'output_parsedLists'.
	These three lists will be used to index parsed terms and phrases.
	"""
	output_parsedLists = []
	output_parsedSingles = []
	output_refIds= []
	for i in range(len(input_raw)):
		output_parsedLists.append([])
		split_raw = re.split(r"[“”  :; ,<>.!?/\|*_ `~*&^%$#@!=()]", input_raw[i])
		for t in split_raw:
			output_parsedLists[i].append(t.lower())
			output_parsedSingles.append(t.lower())
			output_refIds.append(i)
	return output_parsedLists,output_parsedSingles,output_refIds

#run text to search for from the text2search json into the text parser function and set the returned lists to their respective variable names
ss1 = text2search["search_sets"]["architectural_firms"]
parsed_searchSet1 = parseSearchText(ss1)
ss1_lists = parsed_searchSet1[0]
ss1_strings = parsed_searchSet1[1]
ss1_refIds = parsed_searchSet1[2]

ss2 = text2search["search_sets"]["phases_and_terms"]
parsed_searchSet2= parseSearchText(ss2)
ss2_lists = parsed_searchSet2[0]
ss2_strings = parsed_searchSet2[1]
ss2_refIds = parsed_searchSet2[2]

#Scan the existing documents in the output fol
fileNames  = [f for f in listdir(directoryPath_corpus) if isfile(join(directoryPath_corpus, f))]
filesInDirectory = []
for fileName in fileNames:
	split_fileName = fileName.split(".")
	filesInDirectory.append(split_fileName[0])

def keywordStringSearch(split_rawTxt,ss_original,ss_lists,ss_strings,ss_refIds):
	'''This function scans raw text and tests for matches in text2search json.
	Raw input text is parsed and set to lowercase. Each word in the parsed list is itterated over with a for loop.
	If the word appears in 'ss_strings' (list of parsed individual words from the 'parseSearchText' function, 
	the word is indexed and the corresponding list Id values at the corresponding indexes in 'ss_refIds' are taken.
	The list Ids are itterated over and the cooresponding list of individual words from 'ss_lists' are returned.
	The lengt of the returned list of indivudual words is taken and the matched word is indexed. The number of words to
	the left and right are calculated and used to determine the slice size for the slice of words to compare from the raw text.
	If the slice from the raw text and the list of parsed words match, the matched words are returned in the form of a joined string.'''
	
	def get_slice(kwList,txtList,kw):
		idx_kw_kwList = kwList.index(kw)
		idx_kw_txtList = txtList.index(kw)
		np = idx_kw_kwList
		ns = abs(idx_kw_kwList-len(kwList))
		if np <= idx_kw_txtList and ns <= len(txtList):
			idx_p = abs(idx_kw_txtList-np)
			idx_s = abs(idx_kw_txtList+ns)
			slice_txtList = list(txtList[idx_p:idx_s])
			return slice_txtList;
		else:
			return "SKIP"
	matched_keyWordStrings = []
	for txt in split_rawTxt:
		txtLower = txt.lower()
		if txtLower in ss_strings:
			idx_txt = [i for i, t in enumerate(ss_strings) if t == txtLower]
			kwIds = list(dict.fromkeys(list(map(lambda idx: ss_refIds[idx],idx_txt))))
			kwLists = list(map(lambda idx: ss_lists[idx],kwIds))
			kwStrings = list(map(lambda idx: ss_original[idx],kwIds))
			for kwList,kwString in zip(kwLists,kwStrings):
				len_kwList = int(len(kwList))
				txtSlice = get_slice(kwList,split_rawTxt,txtLower)
				if txtSlice != "SKIP":
					match = 0
					for t,kw in zip(txtSlice,kwList):
						#print(t,kw)
						if t == kw:
							match +=1
					if match == len_kwList:
						matched_keyWordStrings.append(kwString)
	return matched_keyWordStrings

def wordStemmer(split_rawTxt,stopwords):
	"""This function stremms raw text if the text is not a stopword"""
	alphabet = [
		"a","b","c","d","e","f","g","h",
		"i","j","k","l","m","n","o","p",
		"q","r","s","t","u","v","w","x","y","z"
		]
	words2stem = []
	for txt in split_rawTxt:
		len_txt = int(len(list(txt)))
		if len_txt >=3:
			if txt.lower() not in stopwords:
				alphaCharCount = 0
				for t in list(txt):
					if t.lower() in alphabet:
						alphaCharCount+=1
				if float(alphaCharCount+1/len_txt+1) >= .75:
					words2stem.append(txt.lower())
	tokens = nltk.word_tokenize(' '.join(words2stem))
	stemWords = [porterStemmer.stem(word) for word in tokens]
	notStopwords = []
	for stem in stemWords:
		if len(list(stem)) >=3:
			if stem not in stopwords:
				notStopwords.append(stem)	
	return notStopwords,words2stem;

class Term_Stats:
	def __init__(self,ts0,ts1,ts2,ts3,ts4,ts5,ts6,ts7,ts8,ts9,ts10,ts11,ts12):
		self.id = ts0
		self.term = ts1
		self.term_original = ts2
		self.df = ts3
		self.docIds = ts4
		self.inDocFreq = ts5
		self.tf = ts6
		self.idf = ts7
		self.tfidf = ts8
		self.mean_tfidf = ts9
		self.mean_tf = ts10
		self.tfidf_normalized = ts11
		self.xy = ts12

class Document_Stats:
	def __init__(self,ds0,ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9,ds10,ds11,ds12,ds13):
		self.id = ds0
		self.id_short = ds1
		self.title = ds2
		self.title_stems = ds3
		self.pubDate = ds4
		self.topic = ds5
		self.termIds = ds6
		self.tf = ds7
		self.x = ds8
		self.y = ds9
		self.high_tfidf_terms = ds10
		self.high_tfidf_termStems = ds11
		self.search_set1 = ds12
		self.search_set2 = ds13

def termStats(stems,origs,statObjs,statObj_doc,statObj_refs,bow,dId_s):
	for ts,to in zip(list(stems),list(origs)):
		if len(bow) == 0:
			bow.append(ts)
			statObj_refs.append(0)
			statObjs[0] = Term_Stats(0,ts,to,0,[dId_s],[1],[1],[],[],0,0,[],[])
			statObj_doc.termIds.append(0)
			statObj_doc.tf.append(1)
		elif len(bow) >= 1:
			if ts not in bow:
				bow.append(ts)
				objId = int(max(statObj_refs)+1)
				statObj_refs.append(objId)
				statObjs[objId] = Term_Stats(objId,ts,to,0,[dId_s],[1],[1],[],[],0,0,[],[])
				statObj_doc.termIds.append(objId)
				statObj_doc.tf.append(1)
			elif ts in bow:
				idx_stem = bow.index(ts)
				statObj = statObjs.get(statObj_refs[idx_stem])
				if docId_short not in TermStatObj.docIds:
					statObj.docIds.append(dId_s)
					statObj.tf.append(1)
					statObj_doc.termIds.append(objId)
					statObj_doc.tf.append(1)
				elif docId_short in statObj.docIds:
					statObj.tf[int(statObj.docIds.index(dId_s))] +=1
					statObj_doc.tf[int(statObj_doc.termIds.index(objId))] +=1

	return statObj,statObj_doc;

DocumentStatObjs = []
DocumentStatObj_refIds = []
bagOfWords = []
TermStatObjs = {}
TermStatObj_refIds = []
store_docIds_short = []
store_docIds_long = []

for file in fileNames:
	filePath = str("%s%s" % (directoryPath_corpus,file))
	fileStats = os.stat(filePath)
	split_filePath = re.split(r"[/.]", str(filePath))
	docTitle = split_filePath[-2]
	docExt = split_filePath[-1]
	try:
		if docExt == "txt":
			docCTime = "_".join(str(fileStats.st_ctime).split("."))
			print(docCTime,docTitle)
			docId_short = ""
			if docCTime in cl_docIds:
				idx_docCTime = int(list(cl_docIds).index(docCTime))
				docId_short = cl_docIds_short[idx_docCTime]
				docPubDate = cl_docPupDate[idx_docCTime]
				docTopic = cl_docTopic[idx_docCTime]
			store_docIds_short.append(docId_short)
			DocStatObj = Document_Stats(docCTime,docId_short,docTitle,[],docPubDate,docTopic,[],[],[],[],[],[],[],[])
			with open(filePath, encoding='utf8') as txtDoc:
				txtDoc_lines = txtDoc.readlines()
				docText = []
				isText = 0
				for line in txtDoc_lines:
					line = line.strip()
					if str(line) == "<TEXT BEGIN>":
						isText = 1
					elif str(line) == "<TEXT END>":
						isText = 0
					if isText == 1 and line not in ["<TEXT END>","<TEXT BEGIN>"]:
						split_rawTxt =  re.split(r"[“”  :;' ,<>.!?/\|*_+ `~*^%$#@!=()]", str(line))
						split_rawTxt_lower = list(map(lambda t: str(t.lower()),split_rawTxt))

						searchSet1_results = keywordStringSearch(split_rawTxt_lower,ss1,ss1_lists,ss1_strings,ss1_refIds)
						if len(searchSet1_results) >=1:
							for i in range(len(searchSet1_results)):
								if searchSet1_results[i] not in DocStatObj.search_set1:
									DocStatObj.search_set1.append(searchSet1_results[i])

						searchSet2_results = keywordStringSearch(split_rawTxt_lower,ss2,ss2_lists,ss2_strings,ss2_refIds)
						if len(searchSet2_results) >=1:
							for i in range(len(searchSet2_results)):
								if searchSet2_results[i] not in DocStatObj.search_set2:
									DocStatObj.search_set2.append(searchSet2_results[i])

						txtStemmer = wordStemmer(split_rawTxt_lower,stopwordList)
						termStems = txtStemmer[0]
						termOriginals = txtStemmer[1]
						termStats(termStems,termOriginals,TermStatObjs,DocStatObj,TermStatObj_refIds,bagOfWords,docId_short)
						
						for termStem,termOriginal in zip(list(termStems),list(termOriginals)):
							if len(bagOfWords) == 0:
								bagOfWords.append(termStem)
								TermStatObj_refIds.append(0)
								TermStatObjs[0] = Term_Stats(0,termStem,termOriginal,0,[docId_short],[1],[1],[],[],0,0,[],[])
								DocStatObj.termIds.append(0)
								DocStatObj.tf.append(1)
							elif len(bagOfWords) >= 1:
								if termStem not in bagOfWords:
									bagOfWords.append(termStem)
									objId = int(max(TermStatObj_refIds)+1)
									TermStatObj_refIds.append(objId)
									TermStatObjs[objId] = Term_Stats(objId,termStem,termOriginal,0,[docId_short],[1],[1],[],[],0,0,[],[])
									DocStatObj.termIds.append(objId)
									DocStatObj.tf.append(1)
								elif termStem in bagOfWords:
									idx_stem = bagOfWords.index(termStem)
									TermStatObj = TermStatObjs.get(TermStatObj_refIds[idx_stem])
									if docId_short not in TermStatObj.docIds:
										TermStatObj.docIds.append(docId_short)
										TermStatObj.tf.append(1)
										DocStatObj.termIds.append(objId)
										DocStatObj.tf.append(1)
									elif docId_short in TermStatObj.docIds:
										TermStatObj.tf[int(TermStatObj.docIds.index(docId_short))] +=1
										DocStatObj.tf[int(DocStatObj.termIds.index(objId))] +=1
										
			DocumentStatObjs.append(DocStatObj)
			DocumentStatObj_refIds.append(docId_short)
			txtDoc.close()

	except Exception as e:
		print(e)
		continue

print(".....")
sorted_bagOfWords = sorted(bagOfWords)
pi = math.pi
def PointsInCircum(r,n):
	return [[math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r] for x in range(0,n+1)]
vectorPoints = PointsInCircum(100,n=int(len(bagOfWords)))
store_tfidf = []
store_meanTfidf = []
store_meanTf = []
store_df = []
for TermStatObj_key, TermStatObj in TermStatObjs.items():
	try:
		N = len(store_docIds_short)
		df = len(TermStatObj.docIds)
		TermStatObj.df = df
		store_df.append(df)
		idf = math.log2(N/df+1)
		point = vectorPoints[sorted_bagOfWords.index(str(TermStatObj.term))]
		TermStatObj.xy = point
		for docId,tf in zip(TermStatObj.docIds,TermStatObj.tf):
			tf_logScale = math.log2(tf+1)
			tfidf = float(tf_logScale*(idf))
			TermStatObj.tfidf.append(tfidf)
			store_tfidf.append(tfidf)
		TermStatObj.mean_tfidf = np.mean(TermStatObj.tfidf)
		TermStatObj.mean_tf = np.mean(TermStatObj.tf)
		store_meanTfidf.append(TermStatObj.mean_tfidf)
		store_meanTf.append(TermStatObj.mean_tf)
	except Exception as e:
		print(e)
		pass

min_tfidf = min(store_tfidf)+1
max_tfidf = max(store_tfidf)+1
delta_tfidf = abs(min_tfidf - max_tfidf)

min_meanTfidf = min(store_meanTfidf)+1
max_meanTfidf = max(store_meanTfidf)+1
delta_minMax_tfidf = abs(min_meanTfidf - max_meanTfidf)

min_tf = min(store_meanTf)+1
max_tf = max(store_meanTf)+1
delta_minMax_tf = abs(min_tf - max_tf)

min_df = min(store_df)+1
max_df = max(store_df)+1
delta_minMax_df = abs(min_df - max_df)

write_termStats = open("%s%s" % (directoryPath_ledgers,"Term Stats.csv"),'w',newline='',encoding='utf-8')
writer_termStats = csv.writer(write_termStats)
writer_termStats.writerow(["term","mean_tfidf","pt_x","pt_y","docIds","docFreq","tfidf"])

for TermStatObj_key,TermStatObj in TermStatObjs.items():
	try:
		min_tfidf = min(TermStatObj.tfidf)+1
		max_tfidf = max(TermStatObj.tfidf)+1
		delta_tfidf = abs(min_tfidf - max_tfidf)
		for docId,tfidf in zip(TermStatObj.docIds,TermStatObj.tfidf):
			norm_tfidf = (tfidf - min_tfidf)/delta_tfidf
			if norm_tfidf >= .75:
				if docId in DocumentStatObj_refIds:
					print(TermStatObj.term_original)
					DocStatObj = DocumentStatObjs[DocumentStatObj_refIds.index(docId)]
					DocStatObj.high_tfidf_terms.append(str(TermStatObj.term_original))
					DocStatObj.high_tfidf_termStems.append(str(TermStatObj.term))

	except Exception as e:
		print(e)
		continue
	try:
		norm_meanTfidf = (TermStatObj.mean_tfidf - min_meanTfidf)/delta_minMax_tfidf
		norm_tf = (TermStatObj.mean_tf - min_tf)/delta_minMax_tf
		norm_df = (TermStatObj.df - min_df)/delta_minMax_df
		rangeMin = .5
		inRange = 0
		if norm_meanTfidf >= rangeMin:
			inRange+=1
		if norm_df >= rangeMin:
			inRange+=1
		writer_termStats.writerow([
			TermStatObj.term,
			inRange,norm_meanTfidf,norm_tf,norm_df,
			point[0],point[1],
			str(";".join(list(map(lambda t: str(t),TermStatObj.docIds))))
			])
	except Exception as e:
		print(e)
		continue

write_docStats = open("%s%s" % (directoryPath_ledgers,"Document Stats.csv"), 'w',newline='',encoding='utf-8')
writer_docStats = csv.writer(write_docStats)
writer_docStats.writerow(["term","mean_tfidf","pt_x","pt_y","docIds","docFreq","tfidf"])

for DocStatObj in DocumentStatObjs:
	try:
		for i in range(len(DocStatObj.termIds)):
			termId = DocStatObj.termIds[i]
			termStatObj = TermStatObjs.get(termId)
			DocStatObj.x.append(termStatObj.xy[0])
			DocStatObj.y.append(termStatObj.xy[1])
		writer_docStats.writerow([
			np.mean(DocStatObj.x),np.mean(DocStatObj.y),
			DocStatObj.id,
			DocStatObj.id_short,
			DocStatObj.title,
			DocStatObj.pubDate,
			DocStatObj.topic,
			";".join(DocStatObj.high_tfidf_terms),
			";".join(DocStatObj.high_tfidf_termStems),
			";".join(DocStatObj.search_set1),
			";".join(DocStatObj.search_set2)
			])
	except Exception as e:
		print(e)
		continue

write_termStats.close()
write_docStats.close()
print("DONE")
