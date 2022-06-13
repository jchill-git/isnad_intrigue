#Given isnads with names marked with @Name_Beg@ and @Name_End@,
# creates tag data for use with NER models
# performs orthographic normalization

import sys
import os
import json
import argparse
import re

from tqdm import tqdm

def normalizeArabicLight(text):
	new_text = text
	new_text = re.sub("[إأٱآا]", "ا", new_text)
	new_text = re.sub("[يى]ء", "ئ", new_text)
	new_text = re.sub("ى", "ي", new_text)
	new_text = re.sub("(ؤ)", "ء", new_text)
	new_text = re.sub("(ئ)", "ء", new_text)
	return new_text

def tokenize(text):
	#remove order markers
	text = text.replace("\u202a","")
	text = text.replace("\u202c","")
	text = text.replace("\u202b","")

	arabicRegex = "[ذ١٢٣٤٥٦٧٨٩٠ّـضصثقفغعهخحجدًٌَُلإإشسيبلاتنمكطٍِلأأـئءؤرلاىةوزظْلآآ]+"
	tagRegex = "@Name_Beg@|@Name_End@"

	#combine the two regexes with a | to get a regex that find both tags and arabic words
	fullRegex = arabicRegex+"|"+tagRegex

	tokens = [m for m in re.finditer(fullRegex,text)]

	tokenStarts = [m.start() for m in tokens]
	tokenEnds = [m.end() for m in tokens]
	tokens = [m.group() for m in tokens]

	return tokens

def extractTags(text):
	tags = []
	taggedTokens = []
	tokens = tokenize(text)

	inGenre =  False
	startingNew = False
	index = 0
	namesFound = 0
	endsFound = 0
	for token in tokens:
		index += 1
		#skip character order markers and new lines
		if token in ["\u202b","\u202c","\u202a","\n"] or len(token)==0:
			continue

		#check if we've started a new genre tagged span
		if token in ["@Name_Beg@"]:
			inGenre = True
			startingNew = True
			namesFound += 1
			# print(isnadsFound)
			# print("@Isnad_Beg@ found at %d"%index)
		elif token in ["@Name_End@"]:
			inGenre = False
			endsFound += 1
			# print(endsFound)
			# print("@Isnad_End@ found at %d"%index)

		if token not in ["@Name_Beg@","@Name_End@"] and len(token)>0:
			if not inGenre:
				tags.append("O")
			elif inGenre and startingNew:
				tags.append("B_PER")
				startingNew = False
			elif inGenre and tags[-1] != "O":
				tags.append("I_PER")

			taggedTokens.append(token)

	return tags,taggedTokens

parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
args = parser.parse_args()

#read in the data
data = [json.loads(s) for s in open(args.infile,"r",encoding="utf8").readlines()]

#print(data[0].keys())

#extract tags from each isnad
outfile = open(args.outfile,"w",encoding="utf8")
for i,d in enumerate(data):
	ID = d["textID"]+"_"+str(i)
	docID = d["textID"]

	#text = normalizeArabicLight(d["text"])
	text = d["text"]

	tags,tokens = extractTags(text)

	outData = {"bookID":docID,"id":ID,"tags":tags,"tokens":tokens,"untaggedText":d["untaggedText"]}

	outfile.write(json.dumps(outData,ensure_ascii=False)+"\n")