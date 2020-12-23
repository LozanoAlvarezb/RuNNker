import os
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

from sentence_transformers import SentenceTransformer


class SentBERT_RNN(torch.nn.Module):

	def __init__(self, args):
		super(SentBERT_RNN,self).__init__()

		self.args = args

		self.P_D = args.pos_dim
		self.P_V = args.pos_num
		self.H = args.hidden_size
		
		P_D = self.P_D
		P_V = self.P_V
		H=self.H
		
		# Load SentenceBERT model from pretrained
		if os.path.isdir(args.save_dir + args.sbert):
			self.sbert = SentenceTransformer(args.save_dir + args.sbert, device=args.device)
		else:
			self.sbert = SentenceTransformer(args.sbert, device=args.device)
			self.sbert.save(args.save_dir + args.sbert)

		self.E = self.sbert.get_sentence_embedding_dimension()
		E=self.E
		

		if args.rnn in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, args.rnn)(
				input_size = E,
				hidden_size = H,
				batch_first = True,
				bidirectional = True
			)

		self.fdoc = nn.Linear(2*H,2*H)

		 # Parameters of Classification Layer
		self.abs_pos_embed = nn.Embedding(P_V,P_D)
		self.content = nn.Linear(2*H,1,bias=False)
		self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
		self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
		self.abs_pos = nn.Linear(P_D,1,bias=False)
		# self.rel_pos = nn.Linear(P_D,1,bias=False)
		self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))

	def forward(self,batch,lengths):
		
		device = self.args.device

		sents_emb = [torch.tensor(
					self.sbert.encode(
						[sent for sent in sents if sent != None],show_progress_bar=False
					)
				) 
				for sents in batch]

		# Padded sequences
		emb = torch.nn.utils.rnn.pad_sequence(sents_emb).to(device) # (max_doc_length,B,E)

		#DEBUG
		if emb.shape[0]>1000:
			print(emb.shape)

		sents = self.rnn(emb)[0] # (max_doc_length,B,2*H)

		probs = []
		
		# Iterate over the batch
		for i in range(0,sents.shape[1]):
			valid_hidden = sents[0:lengths[i],i,:]	# (doc_length,E)
			doc = torch.tanh(self.fdoc(torch.mean(valid_hidden,0))).unsqueeze(0)


			summ = Variable(torch.zeros(1,self.H*2)).to(device)

			# Iterate over posible answers 
			for position, h in enumerate(valid_hidden):
				h = h.view(1, -1)		# (1,2*H)

				# Absolute position in the rank clipped to 1000
				abs_index = Variable(torch.LongTensor([[min(999,position)]])).to(device)
				abs_features = self.abs_pos_embed(abs_index).squeeze(0)
				
				# Relative position in the rank
				# rel_index = int(round((position + 1) * 9.0 / 10000))
				# rel_index = Variable(torch.LongTensor([[rel_index]])).to(device)
				# rel_features = self.rel_pos_embed(rel_index).squeeze(0)
				
				# classification layer
				content = self.content(h) 
				salience = self.salience(h,doc)
				novelty = -1 * self.novelty(h,torch.tanh(summ))
				abs_p = self.abs_pos(abs_features)
				# rel_p = self.rel_pos(rel_features)
				# prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)

				# Class
				prob = torch.sigmoid(content + salience + novelty + abs_p + self.bias)

				summ = summ + torch.mm(prob,h)
				probs.append(prob)
	
		return torch.cat(probs).squeeze()