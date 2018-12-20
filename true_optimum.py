from drift_dataset import DriftDataset

def np_lp_loss(x, y, p):
	return (1 / p) * (1 / batch_size) * np.sum((x - y) ** p)

obj_labels = drift_dataset.np_labels()
obj_data = np.transpose(drift_dataset.np_data())
def objective(W):
	loss = np_lp_loss(obj_labels, np.matmul(W, obj_data), p)
	print(loss)
	return loss