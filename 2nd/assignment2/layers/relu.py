from .base_layer import BaseLayer
import numpy as np

class ReLU(BaseLayer):
	def __init__(self):
		self.cache = None

	def forward(self, 
		input_x: np.ndarray
	):	
		print(input_x.shape)
		## TODO: Implement RELU activation function forward pass
		output = np.maximum(input_x, 0)
		# Store the input in cache, required for backward pass
		self.cache = input_x.copy()
		return output

	def backward(self, dout):
		# Load the input from the cache
		x_temp = self.cache
		## TODO: Calculate gradient for RELU 
		dx = np.where(x_temp > 0, 1, 0)
		return dx
