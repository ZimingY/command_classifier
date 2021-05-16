#!/bin/bash
feature_extract=True
train=True

if [ "$feature_extract" == True ]; then
	echo "feature extraction"
	rm -rf data_exp
	python3 prepare_dataset.py --input dataset --output data_exp
fi 

# LSTM training
if [ "$train" == True ]; then
	echo "training"
	mkdir -p exp
	python main.py --input data_exp --output exp --method mean
fi

