import numpy as np
import torch
from torch.utils import data


class DriftDataset(data.Dataset):

	def __init__(self):
		self.data = []
		self.labels = []
		for i in range(1, 11):
			with open('driftdataset/batch{}.dat'.format(i)) as f:
				for line in f.readlines():
					if line[0] == '1':
						features, label = self._parse_line(line)
						self.data.append(features)
						self.labels.append(label)
	
	def _parse_line(self, line):
		terms = line.split(' ')
		terms = [term for term in terms if term != '\n']
		features = torch.tensor([float(term.split(':')[1]) for term in terms[1:]], dtype=torch.float)
		label = torch.tensor(float(terms[0].split(';')[1]), dtype=torch.float)
		return features, label

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i], self.labels[i]

	def input_size(self):
		return len(self.data[0])

	def np_labels(self):
		return np.array([x.numpy() for x in self.labels])

	def np_data(self):
		return np.array([x.numpy() for x in self.data])