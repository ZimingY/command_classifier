import argparse
from model import Model
from params import *
import torch
from data import AudioDataset
import mlflow

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(args, model, test_loader):
	"""
	"""
	model.eval()
	correct = 0
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader):
			logits = model(data)
			pred = logits.argmax(dim=-1)
			correct += torch.eq(pred, target).sum().item()
	mlflow.log_metric("test accuracy", correct/len(test_loader.dataset))
	print("Test Accuracy: {:.2f}".format(correct/len(test_loader.dataset)))

def main(args):
	model = torch.load(args.model).to(device)
	test_data = AudioDataset(args.data, 'test')
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
						 shuffle=False, num_workers=num_workers)
	test(args, model, test_loader)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, help="input data dir")
	parser.add_argument("--model", type=str, help="model file")
	args = parser.parse_args()
	main(args)
