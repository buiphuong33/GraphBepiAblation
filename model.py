# model.py

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from EGAT import EGAT,AE
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence
class GraphBepi(pl.LightningModule):
    def __init__(
        self,
        dssp_dim=13,
        hidden_dim=256,
        edge_dim=51,
        dropout=0.2,
        lr=1e-4,
        metrics=None,
        result_path=None
    ):
        super().__init__()
        self.metrics=metrics
        self.path=result_path
        self.val_preds, self.val_labels = [], []
        self.test_preds, self.test_labels = [], []
        # loss function
        self.loss_fn=nn.BCELoss()
        # Hyperparameters
    
        self.augment_eps = augment_eps
        self.lr = lr
        self.cls = 1
        bias=False
        self.node_encoder = nn.Linear(dssp_dim, hidden_dim, bias=False)

        self.edge_linear=nn.Sequential(
            nn.Linear(edge_dim,hidden_dim//4, bias=True),
            nn.ELU(),
        )
        self.gat=EGAT(hidden_dim,hidden_dim,hidden_dim//4,dropout)
        
        # output
        self.mlp=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, edge):
        V = pad_sequence(V, batch_first=True, padding_value=0).float()
        mask = V.sum(-1) != 0
        lens = mask.sum(1)

        h_all = []
        for i in range(len(V)):
            x = self.node_encoder(V[i, :lens[i]])
            E = self.edge_linear(edge[i]).permute(2, 0, 1)
            x, _ = self.gat(x, E)
            h_all.append(x)

        h = torch.cat(h_all, 0)
        return self.mlp(h)


    def embed(self, V, edge):
        was_train = self.training
        self.eval()
        with torch.no_grad():
            V = pad_sequence(V, batch_first=True, padding_value=0).float()
            mask = V.sum(-1) != 0
            lens = mask.sum(1)

            outs = []
            for i in range(len(V)):
                x = self.node_encoder(V[i, :lens[i]])
                E = self.edge_linear(edge[i]).permute(2,0,1)
                x, _ = self.gat(x, E)
                outs.append(x)
        if was_train:
            self.train()
        return outs

    def training_step(self, batch, batch_idx): 
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        loss=self.loss_fn(pred,y.float())
        self.log('train_loss', loss.cpu().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result=self.metrics.calc_prc(pred.detach().clone(),y.detach().clone())
            self.log('train_auc', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_prc', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        self.val_preds.append(pred.detach())
        self.val_labels.append(y.detach())
        # log loss theo step (tùy chọn)
        loss = self.loss_fn(pred, y.float())
        self.log('val_step_loss', loss.detach().cpu().item(), on_step=True, on_epoch=False)
        return
    # def validation_epoch_end(self,outputs):
    #     pred,y=[],[]
    #     for i,j in outputs:
    #         pred.append(i)
    #         y.append(j)
    #     pred=torch.cat(pred,0)
    #     y=torch.cat(y,0)
    #     loss=self.loss_fn(pred,y.float())
    #     self.log('val_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
    #     if self.metrics is not None:
    #         result=self.metrics(pred.detach().clone(),y.detach().clone())
    #         self.log('val_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('val_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('val_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('val_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return
        pred = torch.cat(self.val_preds, 0)
        y    = torch.cat(self.val_labels, 0)
        # reset bộ đệm
        self.val_preds.clear(); self.val_labels.clear()

        loss = self.loss_fn(pred, y.float())
        self.log('val_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)

        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('val_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mcc',   result['MCC'],   on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1',    result['F1'],    on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        self.test_preds.append(pred.detach())
        self.test_labels.append(y.detach())
        return
    # def test_epoch_end(self,outputs):
    #     pred,y=[],[]
    #     for i,j in outputs:
    #         pred.append(i)
    #         y.append(j)
    #     pred=torch.cat(pred,0)
    #     y=torch.cat(y,0)
    #     loss=self.loss_fn(pred,y.float())
    #     if self.path:
    #         if not os.path.exists(self.path):
    #             os.system(f'mkdir -p {self.path}')
    #         torch.save({'pred':pred.cpu(),'gt':y.cpu()},f'{self.path}/result.pkl')
    #     if self.metrics is not None:
    #         result=self.metrics(pred.detach().clone(),y.detach().clone())
    #         self.log('test_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
    #         self.log('test_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('test_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('test_recall', result['RECALL'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('test_precision', result['PRECISION'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('test_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('test_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('test_bacc', result['BACC'], on_epoch=True, prog_bar=True, logger=True)
    #         self.log('test_threshold', result['threshold'], on_epoch=True, prog_bar=True, logger=True)
    
    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        pred = torch.cat(self.test_preds, 0)
        y    = torch.cat(self.test_labels, 0)
        # reset bộ đệm
        self.test_preds.clear(); self.test_labels.clear()

        loss = self.loss_fn(pred, y.float())

        if self.path:
            if not os.path.exists(self.path):
                os.system(f'mkdir -p {self.path}')
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')

        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('test_loss',      loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUROC',     result['AUROC'],   on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUPRC',     result['AUPRC'],   on_epoch=True, prog_bar=True, logger=True)
            self.log('test_recall',    result['RECALL'],  on_epoch=True, prog_bar=True, logger=True)
            self.log('test_precision', result['PRECISION'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_f1',        result['F1'],      on_epoch=True, prog_bar=True, logger=True)
            self.log('test_mcc',       result['MCC'],     on_epoch=True, prog_bar=True, logger=True)
            self.log('test_bacc',      result['BACC'],    on_epoch=True, prog_bar=True, logger=True)
            self.log('test_threshold', result['threshold'], on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.9, 0.99), lr=self.lr, weight_decay=1e-5, eps=1e-5)
