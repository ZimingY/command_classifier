# command_classifier

## To run the script
The dataset is downloaded from ref[1] in the paper [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209). And only the numbers are kept in the original dataset. 

Rename this folder into dataset
```bash
├── dataset/
│   ├── zero/
│   ├── one/
:	:
│   ├── nine/
```
Prepare the config file `params.py`

Prepare the dataset by using `feature_extract=True` in `run.sh`

Add `--half_lr` after `python3 prepare_dataset.py` if you want decrease the learning rate on plateau. 

To train a LSTM model, use `train=True` in `run.sh`, the model will be saved as `exp/classifier.pth`. 

Test it by using `test=True` in `run.sh` 

## Result
#### No augmentation, use 300 audios per number
| Name  | Value |
| ------------- | ------------- |
| lr  | 0.003  |
| model_layers  | 3  |
| num_epoch  | 50  |

<img src="figs/params-300.png" width="250px" height="140px"/>
Loss & Accuracy: 0.82:
<p float="left">
  <img src="figs/loss-300.png" width="250px" height="140px"/>
  <img src="figs/acc-300.png" width="250px" height="140px"/>
</p>