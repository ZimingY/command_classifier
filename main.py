import torch
import torch.nn as nn
import torch.nn.functional as F
from data import AudioDataset
from model import Model
import argparse
import mlflow
from params import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
# param_dic = {"batch_size":32, "num_workers":4, "num_epoch":10, "lr":3e-3}
# mlkeys = ["num_epoch", "lr", "model_hidden_size",  "model_layers"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(epoch, model, train_loader, criterion, optimizer, scheduler, log_interval=100):
	"""
	"""
	model.train()
	train_loss = 0
	for i, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		logits = model(data)
		loss = criterion(logits, target)
		train_loss += loss.item()
		loss.backward()
		optimizer.step()
		if i % log_interval == 0:
			mlflow.log_metric("train loss", loss.item())
			mlflow.log_metric("lr", optimizer.param_groups[0]['lr'])
			print("[{}] {}/{} loss: {:.2f} lr: {}".format(epoch, i*data.shape[0], len(train_loader.dataset),
														  loss.item(), optimizer.param_groups[0]['lr']))
	return train_loss/len(train_loader.dataset)


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
	scheduler = None
	train_data = AudioDataset(args.input, 'train')
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
									shuffle=True, num_workers=num_workers)

	val_data = AudioDataset(args.input, 'validation')
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
						 shuffle=False, num_workers=num_workers)

	model = Model(device, method)
	criterion = nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	if args.half_lr:
		print("half learning rate on plateau")
		scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.8, patience=2)

	with mlflow.start_run() as run:
		mlflow.log_param("num_epoch", num_epoch)
		mlflow.log_param("lr", lr)
		mlflow.log_param("model_hidden_size", model_hidden_size)
		mlflow.log_param("model_layers", model_layers)

		for epoch in range(num_epoch):
			train_loss = train(epoch, model, train_loader, criterion, optimizer, scheduler, 5)
			if scheduler:
				scheduler.step(train_loss)
			mlflow.pytorch.log_model(model, "classifier")
			validate(args, model, val_loader, criterion)
		torch.save(model, f'{args.output}/classifier.pth')


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=str, help="input data dir")
	parser.add_argument("--output", type=str, help="output models/logs/ .. dir")
	parser.add_argument("--half_lr", action='store_true')
	args = parser.parse_args()
	main(args)

