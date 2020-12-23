import os
import json

import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, list_IDs):
		'Initialization'
		self.list_IDs = list_IDs

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		# Select sample
		ID = self.list_IDs[index]

		with open('data/'+ID) as json_file:
			data = json.load(json_file)

		# Load data and get label
		X = data['sentences']

		labels = np.zeros(len(X))
		labels[data['labels']] = 1
		y = labels

		return X, y

def pad(DataLoaderBatch):
	"""
	DataLoaderBatch should be a list of (sequence, target, length) tuples...
	Returns a padded tensor of sequences sorted from longest to shortest, 
	"""
	lengths = [len(sample[0]) for sample in DataLoaderBatch]
	max_length = min(1000,max(lengths))
	lengths = [min(1000,length) for length in lengths]

	pad_docs = []
	pad_labels = []
	for i,sample in enumerate(DataLoaderBatch):
		size = min(max_length,lengths[i])
		pad_docs.append(sample[0][0:size] + [None]*(max_length-lengths[i]))
		pad_labels.append(np.append(sample[1][0:size],np.nan*np.zeros((max_length-lengths[i])))) 		
		
	return [pad_docs,torch.tensor(pad_labels,dtype=torch.float)]