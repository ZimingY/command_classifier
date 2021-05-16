import argparse
from pathlib import Path
import shutil, random, os
import librosa
import numpy as np
from params import *

def preprocess_wav(audio, mel_win = mel_win, mel_step = mel_step, mel_channel = mel_channel):
	"""
	return MFCC features of an audio
	"""
	wav, sampling_rate = librosa.load(str(audio), sr=None)
	if sampling_rate != 16000:
		print('audio wrong sr: ', audio)
		exit(1)
	frames = librosa.feature.melspectrogram(
		wav,
		sampling_rate,
		n_fft = int(sampling_rate * mel_win / 1000),
		hop_length = int(sampling_rate * mel_step/1000),
		n_mels = mel_channel)
	return frames.astype(np.float32).T

def add_noise(audio, factor = 1e-4, mel_win = mel_win, mel_step = mel_step, mel_channel = mel_channel):
	wav, sampling_rate = librosa.load(str(audio), sr=None)
	noise = np.random.randn(len(wav))
	augmented_data = wav + factor * noise
	frames = librosa.feature.melspectrogram(
		augmented_data,
		sampling_rate,
		n_fft = int(sampling_rate * mel_win / 1000),
		hop_length = int(sampling_rate * mel_step/1000),
		n_mels = mel_channel)
	return frames.astype(np.float32).T


def change_pitch(audio, mel_win = mel_win, mel_step = mel_step, mel_channel = mel_channel):
	wav, sampling_rate = librosa.load(str(audio), sr=None)
	y1= librosa.effects.pitch_shift(wav, sampling_rate, n_steps=2)
	frames1 = librosa.feature.melspectrogram(
		y1,
		sampling_rate,
		n_fft = int(sampling_rate * mel_win / 1000),
		hop_length = int(sampling_rate * mel_step/1000),
		n_mels = mel_channel)
	return frames1.astype(np.float32).T
	# y2= librosa.effects.pitch_shift(wav, sampling_rate, n_steps=-3)
	# frames2 = librosa.feature.melspectrogram(
	# 	y2,
	# 	sampling_rate,
	# 	n_fft = int(sampling_rate * mel_win / 1000),
	# 	hop_length = int(sampling_rate * mel_step/1000),
	# 	n_mels = mel_channel)
	# return [frames1.astype(np.float32).T, frames2.astype(np.float32).T]




def prepare_data(data_root, out_root, augment):
	"""
	each number: takes 300 to train, 100 to test, 100 to validate
	"""
	out_root.joinpath('train').mkdir(exist_ok = True, parents = True)
	out_root.joinpath('test').mkdir(exist_ok = True, parents = True)
	out_root.joinpath('validation').mkdir(exist_ok = True, parents = True)
	numdirs = list(data_root.glob("*"))

	def create_set(numdir, filenames, split, count, number, start_ind):
		out_dir = out_root.joinpath(split, number)
		out_dir.mkdir(exist_ok = True, parents = True)
		source_file = out_dir.joinpath("_sources.txt").open('w')
		c = 0
		save_list = []
		while c < count:
			if not filenames[start_ind].endswith('wav'):
				start_ind += 1
				continue
			frames = preprocess_wav(numdir.joinpath(filenames[start_ind]))
			if frames.shape[0] == num_frames:
				save_list.append(frames)
				print(number, numdir.joinpath(filenames[start_ind]), file=source_file)
				c += 1
				if augment:
					save_list.append(add_noise(numdir.joinpath(filenames[start_ind])))
					# save_list.append(change_pitch(numdir.joinpath(filenames[start_ind])))
			start_ind += 1
		np.save(out_dir.joinpath(f'{split}.npy'), np.stack(save_list, axis = 0))
		return start_ind


	for numdir in numdirs:
		number = numdir.relative_to(data_root)
		filenames = random.sample(os.listdir(numdir),len(os.listdir(numdir)))
		i = create_set(numdir, filenames, 'train', train_count, number, 0)
		i = create_set(numdir, filenames, 'validation', val_count, number, i)
		i = create_set(numdir, filenames, 'test', test_count, number, i)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", type=Path, help = "data_root foler")
	parser.add_argument("--output", type=Path, help = "output dir")
	parser.add_argument("--augment", action="store_true")
	args = parser.parse_args()
	args.output.mkdir(exist_ok = True, parents = True)
	prepare_data(args.input, args.output, args.augment)
