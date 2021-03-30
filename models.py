from torch import nn
import torch

class RNN(nn.Module):
    def __init__(self, forwards):
        super(RNN, self).__init__()
        self.rnn_src_embeddings = nn.Embedding(2, 1)
        self.forwards = forwards
        self.rnn_dst_embeddings = nn.Embedding(2, 1)
        self.rnn_src = nn.RNN(1,64)
        self.rnn_dst = nn.RNN(1,64)
        self.nn_dst_port = nn.Sequential(nn.Linear(16,16),
                                         nn.ReLU(),
                                         nn.Linear(16,8))
        self.nn_src_port = nn.Sequential(nn.Linear(16,16),
                                         nn.ReLU(),
                                         nn.Linear(16,8))
        self.final = nn.Sequential(nn.Linear(145, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 1))
                                   
    def forward(self, x1, x2, x3):
        x1 = x1.T
        src = x1[0:32]
        dst = x1[32:64]
        if not self.forwards:
            src = torch.flip(src, [0])
            dst = torch.flip(dst, [0])
        src_port = x2[:,:16]
        dst_port = x2[:,16:]
        protocol = x3
        src = src.long()
        dst = dst.long()

        src = self.rnn_src_embeddings(src)
        dst = self.rnn_dst_embeddings(dst)
        rnn_out_src, _ = self.rnn_src(src)
        rnn_out_src = rnn_out_src[-1]
        rnn_out_dst, _ = self.rnn_dst(dst)
        rnn_out_dst = rnn_out_dst[-1]
        
        dst_port = self.nn_dst_port(dst_port)
        src_port = self.nn_src_port(src_port)

        all_features = torch.cat((rnn_out_src, rnn_out_dst, src_port, dst_port, protocol), dim = 1)
        y_pred = self.final(all_features)
        return y_pred

class HsuRNN(nn.Module):
    def __init__(self, forwards):
        super(HsuRNN, self).__init__()
        self.forwards = forwards
        self.rnn_src = nn.LSTM(1,64)
        self.rnn_dst = nn.LSTM(1,64)
        self.nn_dst_port = nn.Sequential(nn.Linear(16,16),
                                         nn.LeakyReLU(negative_slope=0.3), #TensorFlow Default
                                         nn.Linear(16,8),)
        self.nn_src_port = nn.Sequential(nn.Linear(16,16),
                                         nn.LeakyReLU(negative_slope=0.3),
                                         nn.Linear(16,8),)
        self.final = nn.Sequential(nn.Linear(145, 32),
                                   nn.LeakyReLU(negative_slope=0.3),
                                   nn.Linear(32, 32),
                                   nn.LeakyReLU(negative_slope=0.3),
                                   nn.Linear(32, 1),)
                                   
    def forward(self, x1, x2, x3):
        x1 = x1.T
        src = x1[0:32].unsqueeze(-1)
        dst = x1[32:64].unsqueeze(-1)
        if not self.forwards:
            src = torch.flip(src, [0])
            dst = torch.flip(dst, [0])
        src_port = x2[:,:16]
        dst_port = x2[:,16:]
        protocol = x3
        rnn_out_src, _ = self.rnn_src(src)
        rnn_out_src = rnn_out_src[-1]
        rnn_out_dst, _ = self.rnn_dst(dst)
        rnn_out_dst = rnn_out_dst[-1]
        
        dst_port = self.nn_dst_port(dst_port)
        src_port = self.nn_src_port(src_port)

        all_features = torch.cat((rnn_out_src, rnn_out_dst, src_port, dst_port, protocol), dim = 1)
        y_pred = self.final(all_features)
        return y_pred

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.src = nn.Sequential(nn.Linear(32, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 16))
        self.dst = nn.Sequential(nn.Linear(32, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 16))
        self.nn_dst_port = nn.Sequential(nn.Linear(16,16),
                                         nn.ReLU(),
                                         nn.Linear(16,8))
        self.nn_src_port = nn.Sequential(nn.Linear(16,16),
                                         nn.ReLU(),
                                         nn.Linear(16,8))
        self.final = nn.Sequential(nn.Linear(49, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 1))

    def forward(self, x1, x2, x3):
        src = x1[:, 0:32]
        dst = x1[:, 32:64]
        src_port = x2[:,:16]
        dst_port = x2[:,16:]
        protocol = x3
        src = self.src(src)
        dst = self.dst(dst)
            
        dst_port = self.nn_dst_port(dst_port)
        src_port = self.nn_src_port(src_port)

        all_features = torch.cat((src, dst, src_port, dst_port, protocol), dim = 1)
        y_pred = self.final(all_features)
        return y_pred

class HsuAOLRNN(nn.Module):
    def __init__(self, forwards):
        super(HsuAOLRNN, self).__init__()
        self.embeddings = nn.Embedding(44, 64)
        self.lstm = nn.LSTM(64, 256)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 1)
        self.act = nn.LeakyReLU(negative_slope=0.3) #tf default
                                   
    def forward(self, x1, x2):
        x1 = x1.T
        x1 = self.embeddings(x1)
        #hidden = (torch.randn(1, 1, 256), torch.randn(1, 1, 256))  # clean out hidden state
        lstm_out, _ = self.lstm(x1)
        lstm_out = lstm_out[-1]
        combined_lstm = lstm_out
        activated = self.act(self.fc1(combined_lstm))
        y_pred = torch.flatten(self.fc2(activated))
        return y_pred
