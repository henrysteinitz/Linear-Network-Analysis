import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import minimize
import sys
import torch
import torch.utils.data

from drift_dataset import DriftDataset
from linear_net import LinearNet


p = 2
drift_dataset = DriftDataset()
batch_size = len(drift_dataset)
training_data_loader = torch.utils.data.DataLoader(drift_dataset, 
	batch_size=batch_size, shuffle=False)
input_size = drift_dataset.input_size() 

def lp_loss(x, y, p):
	return (1 / p) * (1 / batch_size) * torch.sum((x - y) ** p)

def np_lp_loss(x, y, p):
	return (1 / p) * (1 / batch_size) * np.sum((x - y) ** p)

obj_labels = drift_dataset.np_labels()
obj_data = np.transpose(drift_dataset.np_data())
def objective(W):
	loss = np_lp_loss(obj_labels, np.matmul(W, obj_data), p)
	print(loss)
	return loss

# p=2 min: 665.2053642112809
# sol = minimize(objective, .01 * np.random.rand(1, input_size), method="CG")
# print(sol)
# sys.exit()

model_shapes = {
	'1-layer': (input_size, 1), 
	'2-layer, width 1': (input_size, 1, 1), 
	'2-layer, width 2': (input_size, 2, 1),
	'3-layer, width 1': (input_size, 1, 1, 1), 
	'3-layer, width 2': (input_size, 2, 2, 1),
	'4-layer, width 1': (input_size, 1, 1, 1, 1), 
	'4-layer, width 2': (input_size, 2, 2, 2, 1),
}
model_errors = {
	'1-layer': [], 
	'2-layer, width 1': [], 
	'2-layer, width 2': [],
	'3-layer, width 1': [],
	'3-layer, width 2': [],
	'4-layer, width 1': [],
	'4-layer, width 2': [],
}

with open('errors', 'rb') as f:
	model_errors = pickle.load(f)

learning_rates = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6] 
# [1e-12, 3e-12, 1e-11, 3e-11, 1e-10, 3e-10, 1e-9, 3e-9]

# for model_name, model_shape in model_shapes.items():
# 	print(len(model_errors[model_name]))
# 	print(batch_size)
# 	epochs = 30000
# 	if len(model_errors[model_name]) == epochs // 50:
# 		print('pass')
# 		continue

# 	min_v_loss = -1
# 	for lr in learning_rates:
# 		loss = 0
# 		for k in range(10):
# 			v_model = LinearNet(model_shape)
# 			v_optimizer = torch.optim.SGD(v_model.parameters(), lr=lr)
# 			v_epochs = 200
# 			for j in range(v_epochs):
# 				# We're not interested in generalization, so we validate on all
# 				# of the training data.
# 				for batch in training_data_loader:
# 					v_optimizer.zero_grad()
# 					data, labels = batch
# 					hyps = v_model(data)
# 					train_loss = lp_loss(hyps, labels, p)
# 					train_loss.backward()
# 					v_optimizer.step()
# 			for batch in training_data_loader:
# 				data, labels = batch
# 				hyps = v_model(data)
# 				loss += lp_loss(hyps, labels, p)
# 				print('k: {}, lr: {}, loss: {}'.format(k, lr, loss))
# 		if loss < min_v_loss or min_v_loss < 0:
# 			min_v_loss = loss
# 			best_lr = lr

# 	print(best_lr)
# 	model = LinearNet(model_shape)
# 	optimizer = torch.optim.SGD(model.parameters(), lr=best_lr)
# 	for j in range(epochs):
# 		for batch in training_data_loader: 
# 			# this should run once since we're doing full gd.
# 			optimizer.zero_grad()
# 			data, labels = batch
# 			hyps = model(data)
# 			loss = lp_loss(hyps, labels, p)
# 			loss.backward()
# 			optimizer.step()
# 		error = 0
# 		for batch in training_data_loader:
# 			data, labels = batch
# 			hyps = model(data)
# 			error += lp_loss(hyps, labels, p)
# 		if j % 50 == 0:
# 			model_errors[model_name].append(error)
# 			with open('errors', 'wb') as f:
# 				pickle.dump(model_errors, f)
# 		if j % 100 == 0:
# 			print('epoch {}: {}'.format(j, error))

for model_name, error in model_errors.items():
	if len(error) > 600:
		show_error = error[-600:]
	elif len(error) < 600:
		continue
	else: 
		show_error = error
	plt.plot([50 * x for x in range(len(show_error))], show_error, label=model_name)

plt.legend(['1-layer', 
	'2-layer, width 1', 
	'2-layer, width 2',
	'3-layer, width 1',
	'3-layer, width 2'], loc='upper right')
plt.show()





