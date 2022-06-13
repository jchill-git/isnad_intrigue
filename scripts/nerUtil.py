#This file contain utility functions and classes for ner tagging with 
# transformers' BERT models

import torch

import numpy as np

#represents a dataset of tokens and their tags
class TaggedTokenDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

#this fuction masks out tags for subwords that start after the
# beginning of a word. Why do this? Why not inherit the tag of
# the first piece?
def encode_tags(tags, encodings, tag2id):
	labels = [[tag2id[tag] for tag in doc] for doc in tags]
	encoded_labels = []
	for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
		# create an empty array of -100
		doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
		arr_offset = np.array(doc_offset)

		# set labels whose first offset position is 0 and the second is not 0
		doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
		encoded_labels.append(doc_enc_labels.tolist())

	return encoded_labels