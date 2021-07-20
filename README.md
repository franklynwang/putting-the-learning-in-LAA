# Putting the "Learning" into Learning-Augmented Algorithms for Frequency Estimation (ICML 2021)

This repository consists of the code for the paper [Putting the "Learning" into Learning-Augmented Algorithms for Frequency Estimation](http://proceedings.mlr.press/v139/du21d.html). The code is written using [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

## Table of Contents

- [Putting the "Learning" into Learning-Augmented Algorithms for Frequency Estimation (ICML 2021)](#putting-the-learning-into-learning-augmented-algorithms-for-frequency-estimation-icml-2021)
  - [Table of Contents](#table-of-contents)
  - [Sketches](#sketches)
    - [Obtaining Synthetic Results](#obtaining-synthetic-results)
  - [Data](#data)
  - [Model Components](#model-components)
    - [Model Training](#model-training)
    - [Generating Predictions](#generating-predictions)
    - [Showing coverage plots](#showing-coverage-plots)
    - [Generating Error Ratios](#generating-error-ratios)
    - [Plotting Error Ratios](#plotting-error-ratios)
    - [Plotting Screened Rates](#plotting-screened-rates)
  - [Citation](#citation)

## Sketches

To run the code for this project, you will need the quick_sketches library, which you can find [here](https://github.com/franklynwang/quick-sketches), or simply run:

```
pip install quick_sketches
```

### Obtaining Synthetic Results

To obtain synthetic results, one can simply use `get_synthetic_errors.ipynb`.

## Data

For getting the data, which is not publicly available, we defer to the instructions in Hsu et al's repository, reproduced below.

Dataset website: http://www.caida.org/data/passive/passive_dataset.xml

We preprocessed the number of packets and features for each unique internet flow in each minute (equinix-chicago.dirA.20160121-${minute}00.ports.npy):

```
>>> import numpy as np
>>> data = np.load('equinix-chicago.dirA.20160121-140000.ports.npy').item()
>>> data.keys()
dict_keys(['y', 'x', 'note'])
>>> data['note']
'./data/2016/20160121-130000.UTC/equinix-chicago.dirA.20160121-140000'
>>> data['x'].shape         # 1142626 unique flows in this minute
(1142626, 11)
>>> data['x'][0][:8]        # source ip (first 4) and destination ip (last 4)
array([ 198.,  115.,   14.,  163.,    1.,   91.,  194.,    1.])
>>> data['x'][0][8:10]      # source port and destination port
array([     6.,  35059.])
>>> data['x'][0][-1]        # protocol type
22.0
>>> data['y'][0]            # number of packets
153733
```

Please request data access on the CAIDA website (https://www.caida.org/data/passive/passive_dataset_request.xml). We can share the preprocessed data once you email us the approval from CAIDA (usually takes 2~3 days).

## Model Components

### Model Training

To train the model, run the following command, e.g.

```
python main.py --n-epochs 100 --batch-size 1024 --forwards --split-size 8 \
                        --loss-type "bn" --simple-bn \ #loss type
                        --arch "HsuRNN" \ #architecture of models
                        --checkpoint-path "bn-HsuRNN-8-ckpts-forwards/trial4" \ # path to checkpoint
                        --seed 4 \
                        --train-path ./data/caida/equinix-chicago.dirA.20160121-130000.ports.npy \
                             ./data/caida/equinix-chicago.dirA.20160121-130100.ports.npy \
                             ./data/caida/equinix-chicago.dirA.20160121-130200.ports.npy \
                             ./data/caida/equinix-chicago.dirA.20160121-130300.ports.npy \
                             ./data/caida/equinix-chicago.dirA.20160121-130400.ports.npy \
                             ./data/caida/equinix-chicago.dirA.20160121-130500.ports.npy \
                             ./data/caida/equinix-chicago.dirA.20160121-130600.ports.npy \
                        --valid-path ./data/caida/equinix-chicago.dirA.20160121-130700.ports.npy \
                        --test-path  ./data/caida/equinix-chicago.dirA.20160121-130800.ports.npy \
                        --gpus 1 --auto_select_gpus True &
```

This will save your model under, say, "bn-HsuRNN-8-ckpts-forwards/trial4".

### Generating Predictions

To generate the predictions, run the following command, e.g.

```
python main.py  --evaluate \
                --resume {PATH-TO-CHECKPOINT} \
                 --train-path ./data/caida/equinix-chicago.dirA.20160121-130000.ports.npy \
                         ./data/caida/equinix-chicago.dirA.20160121-130100.ports.npy \
                         ./data/caida/equinix-chicago.dirA.20160121-130200.ports.npy \
                         ./data/caida/equinix-chicago.dirA.20160121-130300.ports.npy \
                         ./data/caida/equinix-chicago.dirA.20160121-130400.ports.npy \
                         ./data/caida/equinix-chicago.dirA.20160121-130500.ports.npy \
                         ./data/caida/equinix-chicago.dirA.20160121-130600.ports.npy \
                 --valid-path ./data/caida/equinix-chicago.dirA.20160121-130700.ports.npy \
                 --test-path  ./data/caida/equinix-chicago.dirA.20160121-130800.ports.npy \
                --generate-predictions \
                --save-name "all_logs/log_mse-HsuRNN-True-ckpts-forwards-more/trial1/predictions08" &
```

which will place the predictions into `all_logs/log_mse-HsuRNN-True-ckpts-forwards-more/trial1/predictions08.npz`.

### Showing coverage plots

To show the screening rates, use `show_screened_rates.ipynb`.

### Generating Error Ratios

To generate the sketch size vs error ratio plots, we use a hyperparameter search in get_results to create a feather file containing our data.

```
python get_results.py --perc-range="np.geomspace(0.2,20,40)" \
                      --nhashes-range 1 2 3 4 \
                      --counter-range 1000 3000 10000 30000 100000 \
                      --ntrials 2 \
                      --sketch-type cm cs \
                      --pred-path tb_logs_modded/log_mse-HsuRNN-True-ckpts-forwards-more/trial1/lightning_logs/predictions08_res.npz \
                      --npy-data equinix-chicago.dirA.20160121-130800.ports.npy \
                      --result-path all_logs/test_results.ftr \
                      --formula std
```

This will save results under result_path.

### Plotting Error Ratios

To plot the results, and to generate the figures, use `plot_ratios.ipynb`

### Plotting Screened Rates

To plot the results, and to generate the figures, use `show_screened_rates.ipynb`

## Citation

If you use our code or cite our paper, please cite

```
@inproceedings{du2021putting,
  title={Putting the “Learning},
  author={Du, Elbert and Wang, Franklyn and Mitzenmacher, Michael},
  booktitle={International Conference on Machine Learning},
  pages={2860--2869},
  year={2021},
  organization={PMLR}
}
```
