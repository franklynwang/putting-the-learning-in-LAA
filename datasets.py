from utils.utils import get_stat, git_log, AverageMeter, keep_latest_files, get_data, get_data_list
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
