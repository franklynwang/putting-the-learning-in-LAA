from utils.utils import get_stat, git_log, AverageMeter, keep_latest_files, get_data, get_data_list
from utils.aol_utils import get_data_aol_feat_list, get_data_aol_by_days

from torch.utils.data import Dataset
import numpy as np

class KeyValDataset(Dataset):
  def __init__(self, npy_files, alpha=1):
    x, y = get_data(npy_files, np.arange(11), 10**8)
    self.query_char_ids = x
    self.log_counts = np.log(y)
    self.sample_weights = np.exp(self.log_counts) ** alpha

  def __getitem__(self, idx):
    return self.query_char_ids[idx][:64].astype(np.float32), self.query_char_ids[idx][64:96].astype(np.float32), self.query_char_ids[idx][-1:].astype(np.float32), self.log_counts[idx]

  def __len__(self):
    return self.log_counts.shape[0]

class AOLDataset(Dataset):
  def __init__(self, npz_files, alpha=1):
    x, y = get_data_aol_feat_list(npz_files)
    self.log_counts = np.log(y)
    self.x = x.astype(np.int64)
    self.lens = x[:,0:1].astype(np.float32)
    self.x = self.x[:,1:]
    self.sample_weights = np.exp(self.log_counts) ** alpha

    
  def __getitem__(self, idx):
    return self.x[idx], self.lens[idx], self.log_counts[idx]

  def __len__(self):
    return self.x.shape[0]

