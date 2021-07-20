import pandas as pd
import numpy as np
import sketches
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
        preds = sketches.count_sketch_preds(nhashes, freqs, width, seed)
    elif sketch_type == "cm":
        preds = sketches.cm_sketch_preds(nhashes, freqs, width, seed)
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
        preds = sketches.count_sketch_preds(nhashes, freqs, width, seed)
    elif sketch_type == "cm":
        preds = sketches.cm_sketch_preds(nhashes, freqs, width, seed)
    preds[samps] = memmed_values[samps]
    space = 4 * width * nhashes + 8 * nbuckets
    return (space, get_f_error(true_out, preds), get_mse(true_out, preds), np.sum(freqs))

def process_error(path, path2, exp_path, formula, path3=None,):
    with torch.no_grad():
        f = np.load(path)
        true = np.load(path2, allow_pickle=True).item()
        true_out = true['y']
        valid_out = f["valid_output"].flatten()
        test_out = f["test_output"].flatten()
        test_out += 0.00001 * np.random.randn(*test_out.shape)
        if path3 is not None:
            f = np.load(path3)
            valid_out_2 = f["valid_output"].flatten()
            test_out_2 = f["test_output"].flatten()
            test_out_2 += 0.00001 * np.random.randn(*test_out_2.shape)
        spaces = []
        f_errors = []
        sums = []
        widths = []
        nhashes_arr = []
        percs = []
        sketch_types = []
        rmses = []
        seeds = []

        tiny_1 = list(itertools.product(np.linspace(2.5, 20, 8), [1], [1000,  10000,  100000], np.arange(100), ["cm", "cs"]))
        tiny_2 = list(itertools.product(np.linspace(2.5, 20, 8), [2, 3, 4], [1000, 10000, 100000], np.arange(20), ["cm", "cs"]))

        qtiny = tiny_1 + tiny_2
        q = qtiny
        with Pool(12) as p:
            if path3 is None:
                if formula == "std":
                    pfunc = partial(get_error, true_out, valid_out, test_out)
                if formula == "ideal":
                    pfunc = partial(get_ideal_error, true_out, valid_out, test_out)    
            if path3 is not None:
                if formula == "std":
                    pfunc = partial(get_error_2, true_out, valid_out, test_out, valid_out_2, test_out_2)
                if formula == "ideal":
                    pfunc = partial(get_ideal_error_2, true_out, valid_out, test_out, valid_out_2, test_out_2)                
            res = p.starmap(pfunc, tqdm(q))
        for i in range(len(q)):
            ele = q[i]
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
def main():
    for formula in ["std"]:
        for minute in [59]:
            for method in ["log_mse-HsuRNN-False-ckpts-forwards-more"]:
                path = f"tb_logs_modded/{method}/trial1/lightning_logs/predictions{minute:02}_res.npz"
                path2 = f"equinix-chicago.dirA.20160121-13{minute:02}00.ports.npy" 
                exp_path = f"tb_logs_modded/{method}/trial1/lightning_logs/minute{minute:02}_{formula}_small_final_results.ftr"
                process_error(path, path2, exp_path, formula)

if __name__ == "__main__":
    main()