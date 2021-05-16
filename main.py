import torch
import torch.nn as nn
import torch.nn.functional as F
from data import AudioDataset
from model import Model
import argparse
import mlflow
from params import *

# param_dic = {"batch_size":32, "num_workers":4, "num_epoch":10, "lr":3e-3}
# mlkeys = ["num_epoch", "lr", "model_hidden_size",  "model_layers"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(epoch, model, train_loader, criterion, optimizer, log_interval=100):
	"""
	"""
	model.train()
	for i, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		logits = model(data)
		loss = criterion(logits, target)
		loss.backward()
		optimizer.step()
		if i % log_interval == 0:
			mlflow.log_metric("train loss", loss.item())
			print("[{}] {}/{} loss: {:.2f}".format(epoch, i*data.shape[0], len(train_loader.dataset), loss.item()))


def validate(args, model, val_loader, criterion):
	"""
	"""
	model.eval()
	loss = 0
	correct = 0
	with torch.no_grad():
		for i, (data, target) in enumerate(val_loader):
			logits = model(data)
			loss += criterion(logits, target).item()
			pred = logits.argmax(dim=-1)
			correct += torch.eq(pred, target).sum().item()
	loss = loss / len(val_loader.dataset)
	mlflow.log_metric("val loss", loss)
	mlflow.log_metric("val accuracy", correct/len(val_loader.dataset))
	print("Validation loss: {:.2f}, accuracy: {:.2f}".format(loss, correct/len(val_loader.dataset)))


def main(args):
	train_data = AudioDataset(args.input, 'train')
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
									shuffle=True, num_workers=num_workers)

	val_data = AudioDataset(args.input, 'validation')
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
						 shuffle=False, num_workers=num_workers)

	model = Model(device, args.method)
	criterion = nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	with mlflow.start_run() as run:
		mlflow.log_param("num_epoch", num_epoch)
		mlflow.log_param("lr", lr)
		mlflow.log_param("model_hidden_size", model_hidden_size)
		mlflow.log_param("model_layers", model_layers)

		for epoch in range(num_epoch):
			train(epoch, model, train_loader, criterion, optimizer, 5)
			mlflow.pytorch.log_model(model, "classifier")
			validate(args, model, val_loader, criterion)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=str, help="input data dir")
	parser.add_argument("--output", type=str, help="output models/logs/ .. dir")
	parser.add_argument("--method", type=str, help="aggregation method for LSTM.")
	args = parser.parse_args()
	main(args)

