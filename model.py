import torch.nn as nn
from params import *

class Model(nn.Module):
	def __init__(self, device, method):
		super().__init__()
		self.hidden_size = 256
		self.method = method
		self.lstm = nn.LSTM(input_size=mel_channel,
							hidden_size=model_hidden_size,
							num_layers=model_layers,
							batch_first=True).to(device)
		self.fc = nn.Sequential(nn.Linear(model_hidden_size, model_hidden_size),
								nn.LeakyReLU(),
								nn.Linear(model_hidden_size, num_classes))
		self.softmax = nn.LogSoftmax(dim=1)
		self.init_weights()

	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.xavier_normal_(module.weight)

	def forward(self, utterances, hidden=None):
		out, (hidden, cell) = self.lstm(utterances, hidden)
		if self.method == 'last':
			out = hidden[-1]
		else:
			out = out.mean(dim=1)
		out = self.fc(out)
		labels = self.softmax(out)
		return labels
