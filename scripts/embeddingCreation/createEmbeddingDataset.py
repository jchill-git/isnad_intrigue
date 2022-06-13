#given a collection of name embeddings and
# their original and disambiguated forms, this script
# will create a dataset in which each name is associated
# with its embedding and arabic and english forms
#
#Author: Ryan Muther, January 2021

import argparse
import re
import json
import sys

import pandas as pd

#read in a csv file of names from isnads
# where each line represents the names
# in a single isnad as a sequence of name entries
def readNameCSV(path,header=True):
	lines = open(path,"r",encoding="utf8").readlines()

	if header:
		start = 1
	else:
		start = 0

	data = []
	isnadNumber = 0
	for l in lines[start:]:
		ID = l.strip().split(",")[0]+"_"+str(isnadNumber)
		#extract the names from the csv (everything past column 3)
		names = [name for name in l.strip().split(",")[4:] if len(name)>0]

		for index,name in enumerate(names):
			d = {}
			d["id"] = ID+"_"+str(index)
			d["docID"] = ID
			d["name"] = name.strip()
			d["disambiguated"] = isEnglish(name)
			data.append(d)
		isnadNumber += 1
	return data

#read in a csv file of isnads
# where each line represents the names
# in a single isnad as a sequence of name entries
def readIsnadCSV(path,header=True):
	lines = open(path,"r",encoding="utf8").readlines()

	if header:
		start = 1
	else:
		start = 0

	data = []
	isnadNumber = 0
	for l in lines[start:]:
		d = {}
		ID = l.strip().split(",")[0]+"_"+str(isnadNumber)
		d["id"] = ID
		#extract the text of each isnad from the line
		text = l.strip().split(",")[1]
		d["text"] = text
		d["tokens"] = text.split(" ")

		#extract the names from the isnad
		d["names"] = [name for name in l.strip().split(",")[4:] if len(name)>0]

		data.append(d)
		isnadNumber += 1
	return data

#return true if name begins with an english character, false otherwise
def isEnglish(name):
	return re.match("[A-Za-z']",name[0]) != None

#returns the start and end index of a name in a given document's tokens
# NB: this will break if a name occurs multiple times. 
# Really only meant for isnads. Even those might be dicey.
def findNames(names,tokens):
	nameEndpoints = []
	startTok = 0
	for name in names:
		nameTokens = [t for t in name.split(" ") if len(t)>0]

		#find all possible starting points for the name (we check if
		# a token is contained rather than just equal to account for
		# removed alephs and such)
		possibleStarts = [i for (i,token) in enumerate(tokens) if nameTokens[0] in token and i>=startTok]

		found = False
		for start in possibleStarts:
			valid = True
			#if we found the name, stop looking
			if found:
				continue
			for i,token in enumerate(nameTokens[1:]):
				if valid and token not in tokens[start+i+1]:
					valid = False
			if valid:
				end = start + len(nameTokens) - 1
				nameEndpoints.append((start,end))
				startTok = end+1
				found = True
		if not found:
			nameEndpoints.append((-1,-1))
	return nameEndpoints

#given an isnad ID and the start and end point of a name,
# determines if the name overlaps with a gold standard span
# returns the required data about who the individual is if so
# returns placeholder data otherwise
def checkForNameOverlap(ID,start,end):
	#get the data for the isnad we're looking at
	possibleNameData = isnadData[ID]

	#get all the named spans in the isnad
	nameSpans = enumerate(possibleNameData["endpoints"])
	namesDisambiguated = possibleNameData["namesDisambiguated"]
	disambiguated = possibleNameData["disambiguated"]
	mentionIDs = possibleNameData["mentionIDs"]
	possibleNames = []
	for i,(nameStart,nameEnd) in nameSpans:
		#this is the more premissive version of assignment
		# where overlapping but not exact spans are allowed
		if not args.exactMatch:
			if (start >= nameStart and start <= nameEnd) or (end >= nameStart and end <= nameEnd):
				possibleNames.append(((nameStart,nameEnd),namesDisambiguated[i],disambiguated[i],mentionIDs[i]))
		#this is the stricter version of assignment
		# where exact spans are required
		else:
			if start == nameStart and end == nameEnd:
				possibleNames.append(((nameStart,nameEnd),namesDisambiguated[i],disambiguated[i],mentionIDs[i]))

	if len(possibleNames)>1:
		print("Overlapping names in gold standard data??")
		print(ID,start,end)
		print(possibleNames)
		sys.exit()

	if len(possibleNames)==0:
		return ((-1,-1),"",False,"")
	else:
		return possibleNames[0]

parser = argparse.ArgumentParser()
parser.add_argument("--fromModelTags",required=False,action="store_true",help="If given, the script will use the provided tags as a base and determine which gold standard mentions they overlap with, rather than assuming correctness")
parser.add_argument("--exactMatch",required=False,action="store_true")
parser.add_argument("--noIbnSad",required=False,action="store_true",help="If provided, produce a dataset without any instances of Ibn Sa'd")
parser.add_argument("arabicNames")
parser.add_argument("disambiguatedNames")
parser.add_argument("embeddings")
parser.add_argument("outfile")
args = parser.parse_args()

#read in data
names = readNameCSV(args.arabicNames)
namesDisambiguated = readNameCSV(args.disambiguatedNames)
nameEmbeddings = pd.read_json(args.embeddings,lines=True,encoding='utf8')
embeddingLen = len(nameEmbeddings["embedding"][0])
nameCount = len(nameEmbeddings)

#convert the name data to dataframes
names = pd.DataFrame(names)
namesDisambiguated = pd.DataFrame(namesDisambiguated)

names = names.drop(columns=["disambiguated"])
namesDisambiguated = namesDisambiguated.rename(columns={"name":"nameDisambiguated"}).drop(columns=["docID"])

#assign individuals to names
if not args.fromModelTags:
	#this is very easy if we're using gold standard tags
	nameEmbeddings = nameEmbeddings.merge(names,on="id")
	nameEmbeddings = nameEmbeddings.merge(namesDisambiguated,on="id")
	#add the filler mention ID field since we'll need a
	# non-trvial one for model evaluation and consistency is nice
	nameEmbeddings["mentionID"] = nameEmbeddings["id"]
	#clean up columns after join
	nameEmbeddings = nameEmbeddings.rename(columns={"docID_x":"docID","name_x":"name"})
	nameEmbeddings = nameEmbeddings.drop(columns=["docID_y","name_y"])
else:
	#this is more complicated if we're using model tags
	#first we need to read in the text of the isnads
	isnads = pd.DataFrame(readIsnadCSV(args.arabicNames))
	isnads["tokens"] = isnads.text.apply(lambda x: x.split(" "))
	#find where each names begins. This has to be done in context due to 
	# possible overlaps in the content of individual names
	isnads["endpoints"] = isnads.apply(lambda x: findNames(x.names,x.tokens),axis=1)
	#add isnad texts to the name data so we can find the token indices
	# at which each name begins
	names = names.merge(isnads,left_on="docID",right_on="id") \
				.drop(columns=["id_y"]) \
				.rename(columns={"id_x":"id"})
	#add the disambiguated names as well so we can
	# assign individuals to names
	names = names.merge(namesDisambiguated,on="id")
	#figure out where each individual name begins by
	# getting the endpoints of the corresponding name
	names["endpoint"] = names.apply(lambda x: x["endpoints"][int(x.id.split("_")[-1])],axis=1)
	#check that we found all the names
	missingNames = names[names.endpoint==(-1,-1)]
	assert(len(missingNames)==0)
	#group the names by isnad ID so we can easily refer to one isnad's names
	groupedByIsnad = names.groupby("docID",as_index=False).agg(lambda x:list(x))
	#collect the data about each isnad's names into a dictionary by ID
	isnadData = {}
	for idx,data in groupedByIsnad.iterrows():
		endpoint = data["endpoint"]
		disambiguated = data["disambiguated"]
		nameList = data["nameDisambiguated"]
		mentionIDs = data["id"]
		isnadData[data["docID"]] = {"endpoints":endpoint,"namesDisambiguated":nameList,"disambiguated":disambiguated,"mentionIDs":mentionIDs}

	#disambiguate the model names by assigning the model names
	# to the names of disambiguated individuals
	# whose spans they overlap with
	nameEmbeddings["disambiguationData"] = nameEmbeddings.apply(lambda x: checkForNameOverlap(x.docID,x.tStart,x.tEnd),axis=1)
	#unpack those results into the colums we expect to see
	nameEmbeddings["nameDisambiguated"] = nameEmbeddings.disambiguationData.apply(lambda x: x[1])
	nameEmbeddings["mentionID"] = nameEmbeddings.disambiguationData.apply(lambda x: x[3])
	nameEmbeddings["disambiguated"] = nameEmbeddings.disambiguationData.apply(lambda x: x[2])
	nameEmbeddings = nameEmbeddings[["bookID","docID","id","name","embedding","nameDisambiguated","mentionID","disambiguated"]]

if args.noIbnSad:
	nameEmbeddings = nameEmbeddings[nameEmbeddings.nameDisambiguated != "Muhammad b. Sa'd"]

#write the name embeddings to a file
print("Writing results to %s"%args.outfile)
outfile = open(args.outfile,"w",encoding="utf8")
for idx,data in nameEmbeddings.iterrows():
	outfile.write(json.dumps(dict(data),ensure_ascii=False)+"\n")