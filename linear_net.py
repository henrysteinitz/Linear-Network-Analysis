import torch
from torch.autograd import Variable
from torch.nn import Module, Parameter


class LinearNet(Module):

	def __init__(self, layer_widths):
		super().__init__()
		self._layers = []
		for i in range(1, len(layer_widths)):
			# Parameters are not added to the parameter list unless they 
			# are attached directly to the model.
			weight_name = "W{}".format(i)
			setattr(self, weight_name, Parameter(
				.001 * torch.randn(layer_widths[i], layer_widths[i-1])))
			self._layers.append(getattr(self, weight_name))

	def forward(self, x):
		# x is a batch with shape = (batch_size, input_size)
		result = torch.transpose(x, 0, 1)
		for layer in self._layers:
			result = torch.matmul(layer, result)
		return result