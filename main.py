import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from pytorch_lightning.metrics import Metric
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from models import HsuRNN, RNN, FFN, HsuAOLRNN
from datasets import KeyValDataset, AOLDataset

# network

class Coverage(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("predictions", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("true", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        self.predictions = torch.cat((self.predictions, preds))
        self.true = torch.cat((self.true, target))

    def compute(self):
        if len(self.predictions) == 0:
          return 1
        idxs = torch.topk(self.predictions, int(len(self.predictions * 1) / 100)).indices
        idxs_ideal = torch.topk(self.predictions, int(len(self.predictions * 1) / 100)).indices
        return torch.exp(self.true).cpu().numpy()[idxs.cpu().numpy()].sum() / torch.exp(self.true).cpu().numpy().sum()

class Regressor(pl.LightningModule):
    def __init__(self, train_path, arch, batch_size, loss_type, weighted=True, alpha=1, forwards=True, simple_bn = False, split_size=64, project_name="CAIDA"):
        super().__init__()
        self.weighted = weighted
        self.forwards = forwards
        self.save_hyperparameters()
        self.project_name = project_name
        
        if loss_type == "bn":
            if simple_bn:
                self.bn = nn.BatchNorm1d(1, affine=False, momentum=0, track_running_stats=False)
            else:
                self.bn = nn.BatchNorm1d(1)
        if arch == "RNN":
            self.model = RNN(forwards=forwards)
        if arch == "HsuRNN":
            self.model = HsuRNN(forwards=forwards)
        if arch == "HsuAOLRNN":
            self.model = HsuAOLRNN(forwards=forwards)
        if arch == "FFN":
            self.model = FFN()
        self.Coverage = Coverage()
     
    def forward(self, x1, x2, x3=None):
        if self.project_name == "CAIDA":
            y_pred = self.model(x1, x2, x3)
        if self.project_name == "AOL":
            y_pred = self.model(x1, x2)
        return y_pred

    def training_step(self, batch, batch_idx):
        if self.project_name == "CAIDA":
            x1, x2, x3, y = batch
            y_pred = self.forward(x1, x2, x3).reshape(-1, 1)
        if self.project_name == "AOL":
            x1, x2, y = batch
            y_pred = self.forward(x1, x2).reshape(-1, 1)
        if self.hparams.loss_type == "bn":
            y_pred_split = torch.split(y_pred, self.hparams.split_size)
            y_score = tuple(self.bn(x) for x in y_pred_split)
            y_score = torch.cat(y_score).type('torch.DoubleTensor').to(self.device)
            y = y.type('torch.DoubleTensor').to(self.device)
            if self.weighted:
                loss = -(y_score.flatten().dot(torch.exp((1 - self.hparams.alpha) * y)))
            else:
                loss = -(y_score.flatten().dot(torch.exp(y)))
        elif self.hparams.loss_type == "log_mse":
            loss = F.mse_loss(y_pred.flatten(), y.flatten())
        elif self.hparams.loss_type == "l1":
            loss = F.l1_loss(torch.exp(y_pred).flatten(), torch.exp(y).flatten())
        self.log('train_loss', loss)
        return loss

    def train_dataloader(self):
        if self.project_name == "CAIDA":
            train_dataset = KeyValDataset(self.hparams.train_path, alpha=self.hparams.alpha)
        elif self.project_name == "AOL":
            train_dataset = AOLDataset(self.hparams.train_path, alpha=self.hparams.alpha)
        else:
            raise NotImplementedError
        train_sampler=None
        if self.hparams.weighted:
            train_sampler = torch.utils.data.WeightedRandomSampler(torch.Tensor(train_dataset.sample_weights).type('torch.DoubleTensor'), 
                                                                len(train_dataset.sample_weights), replacement=True)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle = (train_sampler is None), sampler=train_sampler, pin_memory=True, num_workers=16, drop_last=True)
    
    def validation_step(self, batch, batch_idx):
        if self.project_name == "CAIDA":
            x1, x2, x3, y = batch
            y_pred = self.forward(x1, x2, x3)
        if self.project_name == "AOL":
            x1, x2, y = batch
            y_pred = self.forward(x1, x2)
        self.Coverage(torch.flatten(y_pred), torch.flatten(y))

    def validation_epoch_end(self, outs):
        pred = self.Coverage.predictions.detach().clone()
        true = self.Coverage.true.detach().clone()
        assert (pred.numel() > 0)
        self.Coverage.compute()
        assert (self.Coverage.predictions.numel() == 0)
        true = torch.exp(true) # this gets the exponential of the log counts.
        def comp(pred, true, pct):
            idxs = torch.topk(pred, int(len(pred) * pct / 100)).indices
            return true.cpu().numpy()[idxs.cpu().numpy()].sum() / true.cpu().numpy().sum()
        def ideal(true, pct):
            idxs = torch.topk(true, int(len(pred) * pct / 100)).indices
            return true.cpu().numpy()[idxs.cpu().numpy()].sum() / true.cpu().numpy().sum()
        self.log('coverage@01', comp(pred, true, pct=1))
        self.log('coverage@02', comp(pred, true, pct=2))
        self.log('coverage@05', comp(pred, true, pct=5))
        self.log('coverage@10', comp(pred, true, pct=10))
        self.log('coverage@20', comp(pred, true, pct=20))
        self.log('coverage@30', comp(pred, true, pct=30))
        self.log('coverage@50', comp(pred, true, pct=50))
        self.log('coverage@75', comp(pred, true, pct=75))

        self.log('ideal@01', ideal(true, pct=1))
        self.log('ideal@02', ideal(true, pct=2))
        self.log('ideal@05', ideal(true, pct=5))
        self.log('ideal@10', ideal(true, pct=10))
        self.log('ideal@20', ideal(true, pct=20))
        self.log('ideal@30', ideal(true, pct=30))
        self.log('ideal@50', ideal(true, pct=50))
        self.log('ideal@75', ideal(true, pct=75))
    
    def test_step(self, batch, batch_idx):
        if self.project_name == "CAIDA":
            x1, x2, x3, y = batch
            y_pred = self.forward(x1, x2, x3)
        if self.project_name == "AOL":
            x1, x2, y = batch
            y_pred = self.forward(x1, x2)
        self.Coverage(torch.flatten(y_pred), y)
    
    def test_epoch_end(self, outs):
        pred = self.Coverage.predictions.detach().clone()
        true = self.Coverage.true.detach().clone()
        true = torch.exp(true)
        self.Coverage.compute()

        def comp(pred, true, pct):
            idxs = torch.topk(pred, int(len(pred) * pct / 100)).indices
            return true.cpu().numpy()[idxs.cpu().numpy()].sum() / true.cpu().numpy().sum()
        def ideal(true, pct):
            idxs = torch.topk(true, int(len(pred) * pct / 100)).indices
            return true.cpu().numpy()[idxs.cpu().numpy()].sum() / true.cpu().numpy().sum()
        self.log('test_coverage@01', comp(pred, true, pct=1))
        self.log('test_coverage@02', comp(pred, true, pct=2))
        self.log('test_coverage@05', comp(pred, true, pct=5))
        self.log('test_coverage@10', comp(pred, true, pct=10))
        self.log('test_coverage@20', comp(pred, true, pct=20))
        self.log('test_coverage@30', comp(pred, true, pct=30))
        self.log('test_coverage@50', comp(pred, true, pct=50))
        self.log('test_coverage@75', comp(pred, true, pct=75))

        self.log('test_ideal@01', ideal(true, pct=1))
        self.log('test_ideal@02', ideal(true, pct=2))
        self.log('test_ideal@05', ideal(true, pct=5))
        self.log('test_ideal@10', ideal(true, pct=10))
        self.log('test_ideal@20', ideal(true, pct=20))
        self.log('test_ideal@30', ideal(true, pct=30))
        self.log('test_ideal@50', ideal(true, pct=50))
        self.log('test_ideal@75', ideal(true, pct=75))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--arch", type=str, help="Architecture", default="FFN")
        parser.add_argument("--train-path", type=str, nargs='*', help="training data (.npy file)", default="")
        parser.add_argument('--weighted', action='store_true',
                    help='Whether or not to use our weighting trick')
        parser.add_argument('--loss-type', choices=["bn", "log_mse", "l1"],
                    help='What loss function to use.')
        parser.add_argument('--alpha', type=float, default=1,
                    help='How much to downweight the distribution')
        parser.add_argument('--batch-size', type=int, default=256,
                    help='Batch Size')
        parser.add_argument('--forwards', action='store_true',
                    help='Whether to keep the IP in forward order (default false)')
        parser.add_argument('--simple-bn', action="store_true",
                    help='Simple batchnorm (just mean, variance)')
        parser.add_argument('--split-size', type=int, default=64,
                    help='Size of Splits')
        return parser

import wandb
from pytorch_lightning.loggers import WandbLogger

def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--valid-path", type=str, nargs='*', help="validation data (.npy file)", default="")
    parser.add_argument("--test-path", type=str, nargs='*', help="testing data (.npy file)", default="")
    parser.add_argument('--evaluate', action='store_true',
                        help='Whether to evaluate or not')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to load it from')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to Checkpoints')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--smoke', action='store_true',
                        help='Run single step')
    parser.add_argument('--save-name', type=str, default=None,
                        help='Where to save the results of the experiments')
    parser.add_argument('--generate-predictions', action="store_true",
                        help='Whether or not to generate predictions or simply report metrics.')
    parser.add_argument('--test-batch-size', type=int, default=4096,
                        help='Size of Batch to use when generating predictions')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random Seed')
    parser.add_argument('--project-name', type=str, default="CAIDA",
                        help='Project Name')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Regressor.add_model_specific_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    if args.project_name == "CAIDA":
        test_dataset = KeyValDataset(args.test_path)
    elif args.project_name == "AOL":
        test_dataset = AOLDataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=30)
    if args.valid_path:
        if args.project_name == "CAIDA":
            val_dataset = KeyValDataset(args.valid_path)
        elif args.project_name == "AOL":
            val_dataset = AOLDataset(args.valid_path)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True)
    if args.evaluate and not args.generate_predictions:
        wandb_logger = WandbLogger(project=args.project_name, name=f"{args.resume}-eval")
        pretrained_model = Regressor.load_from_checkpoint(args.resume)
        pretrained_model.freeze()
        trainer = pl.Trainer(gpus=args.gpus, log_every_n_steps=50)
        trainer.test(pretrained_model, test_loader)
    elif args.evaluate and args.generate_predictions:
        wandb_logger = WandbLogger(project=args.project_name, name=f"{args.resume}-genpred")
        pretrained_model = Regressor.load_from_checkpoint(args.resume, train_path=args.train_path,
                arch=args.arch,
                batch_size=args.batch_size,
                loss_type = args.loss_type,
                weighted=args.weighted,
                alpha=args.alpha,
                forwards=args.forwards,
                simple_bn=args.simple_bn,
                split_size=args.split_size,
                project_name=args.project_name)

        pretrained_model.freeze()
        trainer = pl.Trainer(gpus=args.gpus, log_every_n_steps=50)
        trainer.test(pretrained_model, test_loader)
        print(pretrained_model.device)
        val_loss_all = np.array([])
        val_output_all = np.array([])
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=30)
        if args.project_name == "CAIDA":
            for (a, b, c, y) in tqdm(val_loader):
                val_output = pretrained_model(a, b, c)
                print(val_output.shape)
                val_output_all = np.concatenate((val_output_all, val_output.flatten().numpy()))
                val_loss = (val_output.flatten() - y) ** 2
                val_loss_all = np.concatenate((val_loss_all, val_loss.flatten().numpy()))
        if args.project_name == "AOL":
            for (a, b, y) in tqdm(val_loader):
                a = a.to(pretrained_model.device)
                b = b.to(pretrained_model.device)
                val_output = pretrained_model(a, b).cpu()
                print(val_output.shape)
                val_output_all = np.concatenate((val_output_all, val_output.flatten().numpy()))
                val_loss = (val_output.flatten() - y) ** 2
                val_loss_all = np.concatenate((val_loss_all, val_loss.flatten().numpy()))

        assert len(val_loss_all) == len(val_dataset) and len(val_output_all) == len(val_dataset)
        test_loss_all = np.array([])
        test_output_all = np.array([])
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=30)
        if args.project_name == "CAIDA":
            for (a, b, c, y) in tqdm(test_loader):
                test_output = pretrained_model(a, b, c)
                test_loss = (test_output.flatten() - y) ** 2
                print(test_loss.flatten().numpy().shape)
                test_loss_all = np.concatenate((test_loss_all, test_loss.flatten().numpy()))
                test_output_all = np.concatenate((test_output_all, test_output.flatten().numpy()))
                print(len(test_output_all))
        if args.project_name == "AOL":
            for (a, b, y) in tqdm(test_loader):
                a = a.to(pretrained_model.device)
                b = b.to(pretrained_model.device)
                test_output = pretrained_model(a, b).cpu()
                test_loss = (test_output.flatten() - y) ** 2
                print(test_loss.flatten().numpy().shape)
                test_loss_all = np.concatenate((test_loss_all, test_loss.flatten().numpy()))
                test_output_all = np.concatenate((test_output_all, test_output.flatten().numpy()))
                print(len(test_output_all))
        assert len(test_loss_all) == len(test_dataset) and len(test_output_all) == len(test_dataset)
        np.savez(args.save_name+'_res',
            test_output=test_output_all,
            test_loss=test_loss_all,
            valid_output=val_output_all,
        )
    else:
        wandb_logger = WandbLogger(project=args.project_name, name=f"{args.checkpoint_path}")

        if args.resume:
            vanilla_regressor = Regressor.load_from_checkpoint(args.resume, train_path=args.train_path,
                arch=args.arch,
                batch_size=args.batch_size,
                loss_type = args.loss_type,
                weighted=args.weighted,
                alpha=args.alpha,
                forwards=args.forwards,
                simple_bn=args.simple_bn,
                split_size=args.split_size,
                project_name=args.project_name)
        else:
            vanilla_regressor = Regressor(
                train_path=args.train_path, 
                arch=args.arch,
                batch_size=args.batch_size,
                loss_type = args.loss_type, 
                weighted=args.weighted,
                alpha=args.alpha,
                forwards=args.forwards, 
                simple_bn=args.simple_bn,
                split_size=args.split_size,
                project_name=args.project_name,
            )
        early_stop_callback = EarlyStopping(
           monitor='coverage@01',
           min_delta=0.00,
           patience=50,
           verbose=False,
           mode='min'
        )
        trainer = pl.Trainer(default_root_dir=args.checkpoint_path, fast_dev_run=args.smoke, gpus=args.gpus, max_epochs=args.n_epochs, log_every_n_steps=50, logger=wandb_logger, callbacks=[early_stop_callback])
        
        trainer.fit(vanilla_regressor, val_dataloaders=val_loader)
        trainer.test(test_dataloaders=test_loader)

if __name__ == '__main__':
    cli_main()
