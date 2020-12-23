import os, argparse, logging
import math

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_ 
from tqdm import tqdm

import models
from data import Dataset, pad

# max_memory =torch.cuda.max();print(max_memory)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def train(args):

	data = Dataset(os.listdir(args.data_path))

	n_data = len(data)

	train_val, test = torch.utils.data.random_split(data,[
		math.floor(n_data*0.8),
		math.ceil(n_data*0.2)
	])

	n_train = len(train_val)

	training, validation = torch.utils.data.random_split(train_val,[
		math.floor(n_train*0.8),
		math.ceil(n_train*0.2)
	])

	print('#Training Data: %d' % (len(training)))

	# Parameters
	params = {'batch_size': args.batch_size,
			'shuffle': True,
			'num_workers': 2,
			'collate_fn': pad}

	# Generators
	training_generator = torch.utils.data.DataLoader(training, **params)
	validation_generator = torch.utils.data.DataLoader(validation, **params)

	net = getattr(models,args.model)(args)
	net.to(args.device)
	# print("Net GPU usage: {}".format(sizeof_fmt(torch.cuda.memory_allocated())))

	cost = nn.BCELoss()

	# print(net)
	params = sum(p.numel() for p in list(net.parameters())) / 1e6
	print('#Params: %.1fM' % (params))

	min_val_loss = float('inf')
	min_train_loss = float('inf')
	optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
	net.train()

	# Loop over epochs
	for epoch in range(args.epochs):
		
		# Training
		pbar = tqdm(total=len(training),desc="Net GPU usage: {}".format(sizeof_fmt(torch.cuda.memory_allocated())))
		for i,(batch, targets) in enumerate(training_generator):
			
			# Transfer to GPU
			# batch, targets = batch.to(device), targets.to(device);

			# Length of each document
			lengths = [len([sent for sent in sents if sent!=None]) for sents in batch]
			targets = torch.cat([labels[0:lengths[i]] for i,labels in enumerate(targets)])
			targets = targets.to(args.device)

			# Estimate the probability of including each answer
			estimate = net(batch,lengths)

			# Compute the loss
			assert targets.shape == estimate.shape
			loss = cost(estimate,targets)

			# Backprop the loss
			optimizer.zero_grad()
			loss.backward()
			clip_grad_norm_(net.parameters(), args.max_norm)
			optimizer.step()

						# If new best loss save
			if(loss<min_train_loss):
				print()
				logging.info('Epoch: %d\t\t New Min_Train_Loss: %f !!'
							% (epoch,loss))
				min_train_loss = loss.item()
				# print("Total GPU usage pre save: {}".format(sizeof_fmt(torch.cuda.memory_allocated())))
				# net = net.cpu()
				torch.save({
				'epoch': epoch,
				'model_state_dict': net.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss
				}, args.save_dir+args.save+"model_checkpoint.pt")
				# net = net.to(args.device)


			pbar.set_description("Max GPU usage: {}".format(sizeof_fmt(torch.cuda.max_memory_reserved())))
			pbar.update(32)
		pbar.close()
			
		# Validation
		total_loss = 0
		with torch.no_grad():
			for i, (batch, targets) in enumerate(validation_generator):
				# Transfer to GPU
				# Length of each document
				lengths = [len([sent for sent in sents if sent!=None]) for sents in batch]
				targets = torch.cat([labels[0:lengths[i]] for i,labels in enumerate(targets)])
				targets = targets.to(args.device)

				# Estimate the probability of including each answer
				estimate = net(batch,lengths)

				# Compute the loss
				total_loss += cost(estimate,targets).item()
			loss = total_loss / i
		
			if loss < min_val_loss:
				min_val_loss = loss
				logging.info('Epoch: %d\t\t New Min_Val_Loss: %f !!'
							% (epoch,loss))
				torch.save(net, args.save_dir+args.save+"model_loss=%f.pt" % min_val_loss)
				
		

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format='\n%(asctime)s [INFO] %(message)s')
	parser = argparse.ArgumentParser(description='extractive summary applied to answer reranking')

	# model
	parser.add_argument('-save_dir',type=str,default='checkpoints/')
	parser.add_argument('-save',type=str,default='runnker/')
	parser.add_argument('-pos_dim',type=int,default=50)
	parser.add_argument('-pos_num',type=int,default=1000)
	parser.add_argument('-model',type=str,default='SentBERT_RNN')
	parser.add_argument('-rnn', type=str,default='LSTM')
	parser.add_argument('-sbert', type=str, default='gsarti/scibert-nli')
	parser.add_argument('-hidden_size',type=int,default=200)

	# train
	parser.add_argument('-data_path',type=str,default='data/')
	parser.add_argument('-lr',type=float,default=1e-3)
	parser.add_argument('-batch_size',type=int,default=32)
	parser.add_argument('-epochs',type=int,default=5)
	parser.add_argument('-seed',type=int,default=1)
	parser.add_argument('-report_every',type=int,default=1500)
	parser.add_argument('-max_norm',type=float,default=1.0)

	# # test
	# parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
	# parser.add_argument('-test_dir',type=str,default='data/test.json')
	# parser.add_argument('-ref',type=str,default='outputs/ref')
	# parser.add_argument('-hyp',type=str,default='outputs/hyp')
	# parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
	# parser.add_argument('-topk',type=int,default=15)

	# device
	parser.add_argument('-device',type=str, default='cuda')

	# option
	parser.add_argument('-debug',action='store_true')
	# parser.add_argument('-show_encode,',action='store_true')

	args = parser.parse_args()

	# CUDA for PyTorch
	args.device = 'cuda' if (torch.cuda.is_available() and args.device=='cuda') else 'cpu'

	train(args)



