import pandas as pd
import numpy as np
import quick_sketches
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import itertools
from multiprocessing import Pool
from functools import partial
import multiprocessing

def get_f_error(true, pred):
    return np.sum(true * abs(pred - true)) / np.sum(true)

def get_mse(true, pred):
    return np.sqrt(np.sum((pred - true) * (pred - true)) / np.sum(true))

def get_error(true_out, valid_out, test_out, mem_perc, nhashes, width, seed, sketch_type):
    freqs = true_out.copy()
    #print(true_out.sum())
    nbuckets = int((mem_perc) * len(valid_out) / 100)
    phantom_buckets = int((mem_perc) * 1.1 * len(valid_out) / 100)
    assert nbuckets < len(valid_out)
    cutoff = np.partition(valid_out, -phantom_buckets)[-phantom_buckets] #deliberately underset the cutoff.
    test_out_mask = (test_out > cutoff)
    filtered_values = freqs.copy()
    filtered_values[~test_out_mask] = 0
    nsamples = min(np.sum(test_out_mask), nbuckets)
    if nsamples > 0:
        samps = torch.multinomial(torch.Tensor(filtered_values), nsamples)
    memmed_values = true_out.copy()
    if nsamples > 0:
        freqs[samps] = 0
    if sketch_type == "cs":
        preds = quick_sketches.count_sketch_preds(nhashes, freqs, width, seed)
    elif sketch_type == "cm":
        preds = quick_sketches.cm_sketch_preds(nhashes, freqs, width, seed)
    if nsamples > 0:
        preds[samps] = memmed_values[samps]
    space = 4 * width * nhashes + 8 * nbuckets
    return space, get_f_error(true_out, preds), get_mse(true_out, preds), np.sum(freqs)

def get_ideal_error(true_out, valid_out, test_out, mem_perc, nhashes, width, seed, sketch_type):
    freqs = true_out.copy()
    nbuckets = int(mem_perc * len(test_out) / 100)
    cutoff = np.partition(test_out, -nbuckets)[-nbuckets]
    test_out_mask = (test_out > cutoff)
    filtered_values = freqs.copy()
    filtered_values[~test_out_mask] = 0
    samps = torch.multinomial(torch.Tensor(filtered_values), min(np.sum(test_out_mask), nbuckets))
    memmed_values = true_out.copy()
    freqs[samps] = 0
    if sketch_type == "cs":
        preds = quick_sketches.count_sketch_preds(nhashes, freqs, width, seed)
    elif sketch_type == "cm":
        preds = quick_sketches.cm_sketch_preds(nhashes, freqs, width, seed)
    preds[samps] = memmed_values[samps]
    space = 4 * width * nhashes + 8 * nbuckets
    return (space, get_f_error(true_out, preds), get_mse(true_out, preds), np.sum(freqs))

def process_error(path, path2, exp_path, formula,
                  perc_range=np.geomspace(0.2,20,40),
                  nhashes_range=[1,2,3,4],
                  counter_range=[1000,3000,10000,30000,100000],
                  ntrials=100, 
                  sketch_choices=["cm","cs"]):
    with torch.no_grad():
        f = np.load(path)
        true = np.load(path2, allow_pickle=True).item()
        true_out = true['y']
        valid_out = f["valid_output"].flatten()
        test_out = f["test_output"].flatten()
        test_out += 0.00001 * np.random.randn(*test_out.shape)
        spaces = []
        f_errors = []
        sums = []
        widths = []
        nhashes_arr = []
        percs = []
        sketch_types = []
        rmses = []
        seeds = []

        grid = list(itertools.product(perc_range, nhashes_range, counter_range, np.arange(ntrials), sketch_choices))
        with Pool(multiprocessing.cpu_count()) as p:
            if formula == "std":
                pfunc = partial(get_error, true_out, valid_out, test_out)
            if formula == "ideal":
                pfunc = partial(get_ideal_error, true_out, valid_out, test_out)                  
            res = p.starmap(pfunc, tqdm(grid))
        for i in range(len(grid)):
            ele = grid[i]
            result = res[i]
            percs.append(ele[0])
            nhashes_arr.append(ele[1])
            widths.append(ele[2])
            seeds.append(ele[3])
            sketch_types.append(ele[4])
            spaces.append(result[0])
            f_errors.append(result[1])
            rmses.append(result[2])
            sums.append(result[3])
        df = pd.DataFrame({"space": spaces, "f_error": f_errors, "sum": sums, 
                           "width": widths, "nhashes": nhashes_arr, "rmse": rmses, "seed": seeds, 
                           "perc": percs, "sketch": sketch_types})
        df.to_feather(exp_path)

import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perc-range", type=str, default='np.geomspace(0.2,20,40)')
    parser.add_argument("--nhashes-range", nargs='+', required=True)
    parser.add_argument("--counter-range", nargs='+', required=True)
    parser.add_argument("--ntrials", type=int)
    parser.add_argument("--sketch-type", nargs='+')
    parser.add_argument("--pred-path", type=str, required=True, help="Path to predictions from model") 
    parser.add_argument("--npy-data", type=str, required=True, help="Path to ground truth .npy files")
    parser.add_argument("--result-path", type=str, required=True, help="Where to save the results")
    parser.add_argument("--formula", type=str, required=True, help="std (for the 10\% overprediction correction), ideal (for just highest scores)")

    args = parser.parse_args()
    args.perc_range = eval(args.perc_range)
    args.counter_range = list(map(int, args.counter_range))
    args.nhashes_range = list(map(int, args.nhashes_range))

    process_error(args.pred_path, args.npy_data, args.result_path, args.formula, args.perc_range, args.nhashes_range, args.counter_range, args.ntrials, args.sketch_type)
if __name__ == "__main__":
    main()