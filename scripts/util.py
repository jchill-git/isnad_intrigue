#This file contain utility functions and classes for mlm training with 
# transformers' BERT models

import torch
from torch import nn

import numpy as np

#represents a dataset of tokens and their tags
class Dataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)
		
#given tags, extracts the tagged span locations
def extractTaggedSpans(tags):
	i = 0
	taggedSpans = []
	inTaggedSpan = False
	begin = -1
	end = -1
	for i in range(len(tags)):
		tag = tags[i]
		if not inTaggedSpan and tag != "O" and "PAD" not in tag:
			inTaggedSpan = True
			#If we started a span, say the model predicted a B tag
			begin = i
		elif inTaggedSpan and tag == "O":
			inTaggedSpan = False
			end = i-1
			taggedSpans.append((begin,end))
		elif inTaggedSpan and "B" in tag and "PAD" not in tag:
			inTaggedSpan = True
			end = i-1
			taggedSpans.append((begin,end))
			begin = i
		elif inTaggedSpan and i == len(tags)-1:
			end = len(tags)-1
			taggedSpans.append((begin,end))
	return taggedSpans

#Implements a feedforward network roughly along the lines of 
# (Kenton 2019)'s end to end neural coref work.
# two hidden layers w/dropout and ReLU activation
class FFNN(torch.nn.Module):
	def __init__(self,input_size,hidden_size=150):
		"""
		Here we instantiate the various layers of the net and dropout
		 This is probably the form of network I want?
		"""
		super(FFNN, self).__init__()
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, 1)

		#define activations and dropout
		self.dropout = nn.Dropout(0.5)
		self.act = nn.ReLU()
		

	def forward(self, x):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.

		Returns the logits (well, logit in the binary case) the class likelihoods
		"""
		# Apply dropout
		x = self.dropout(x)
		#linear layer 1
		x = self.act(self.linear1(x))
		# Apply dropout
		x = self.dropout(x)
		#linear layer 2
		x = self.act(self.linear2(x))
		# Apply dropout
		x = self.dropout(x)
		#final linear layer
		logits = self.linear3(x)
		return logits.squeeze(1)