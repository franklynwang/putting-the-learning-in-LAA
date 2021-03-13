# Putting the "Learning" into Learning-Augmented Algorithms for Frequency Estimation

This repository consists of the code for the paper "Putting the ``Learning" into Learning-Augmented Algorithms for Frequency Estimation". The code is written using Pytorch Lightning

## Table of Contents

1. Sketches

   a. Installation

   b. Obtaining Synthetic Results

2. Data
3. Model Components

   a. Model Training

   b. Model Evaluation

   c. Generating Predictions

   d. Showing Coverage Plots

   e. Generating Error Ratios

   f. Plotting Error Ratios

## Sketches

We include a Python Package that contains Python bindings to a C++ library to calculate the results of the count-min sketch and count-sketches.

NOTE: this only works on Mac in our current version.

First, you will need pybind11, so run

`pip install pybind11`

Then, run

`sh build.sh` to install the sketches library. This library implements two functions:

```
[numpy array of long longs] cm_sketch_preds(int nhashes, [numpy array of long longs] np_input, ll width, int seed)
```

takes as input a random seed, the number of hashes, the width of the sketch (or the number of cells in each row), and the frequencies of each key. Then, it outputs what a count-min sketch would give as predicted frequencies with that particular set of parameters.

The function `count_sketch_preds` does the analagous function for count-sketches.

### Obtaining Synthetic Results

To obtain synthetic results, one can simply use `get_synthetic_errors.ipynb`.

## Data

For getting the data, which is not publically available, we defer to the instructions in Hsu et al's repository.

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
                        --loss-type "bn" --simple-bn \
                        --arch "HsuRNN" --checkpoint-path "bn-HsuRNN-8-ckpts-forwards/trial4" --seed 4 \
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

### Model Evaluation

To evaluate the model, run the following command, e.g.

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
                        --save-name "l1-HsuRNN-False-ckpts-forwards-more/trial4/predictions08" --gpus 1 --auto_select_gpus True
```

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
which will place the predictions into  `all_logs/log_mse-HsuRNN-True-ckpts-forwards-more/trial1/predictions08.npz`.

### Showing coverage plots

To show the screening rates, use `show_screened_rates.ipynb`. Unfortunately, due to constraints on the size of the supplementary material, we are unable to provide thefiles containing the predictions. 

### Generating Eror Ratios

To generate the sketch size vs error ratio plots, we use a hyperparameter search in get_results to create a feather file containing our data.

`python get_results.py`

This will save results under `all_logs`. Our results are already located there.

### Plotting Error Ratios

To plot the results, and to generate the figures, use `plot_ratios.ipynb`
