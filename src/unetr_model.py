import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex, Dice
from torch.optim.lr_scheduler import CosineAnnealingLR

from .losses import DiceBCELoss, DiceFocalLoss, DiceCELoss


# ------------------------- LightningModule -------------------------
class UNETRModule(pl.LightningModule):
    def __init__(self, model, lr=1e-4, loss_type='bce'):
        super().__init__()
        self.model = model
        self.lr = lr

        # Choose loss
        if loss_type == 'bce':
            self.loss_fn = DiceBCELoss()
        elif loss_type == 'ce':
            self.loss_fn = DiceCELoss()
        elif loss_type == 'focal':
            self.loss_fn = DiceFocalLoss()
        else:
            raise ValueError("loss_type must be 'bce' or 'focal'")

        # Metrics
        self.dice_metric = Dice(num_classes=3, average='macro')
        self.iou_metric = MulticlassJaccardIndex(num_classes=3, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['pixel_values'], batch['labels']  # (B,3,H,W), (B,H,W)
        logits = self(x) 
        y = y.squeeze(1)                              # (B,2,H,W)
        # loss = self.loss_fn(logits, torch.nn.functional.one_hot(y, 2).permute(0, 3, 1, 2).float())
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['pixel_values'], batch['labels']
        logits = self(x)
        y = y.squeeze(1)#.long()
        preds = torch.argmax(logits, dim=1)
        self.dice_metric.update(preds, y)
        self.iou_metric.update(preds, y)

    def on_validation_epoch_end(self):
        dice = self.dice_metric.compute()
        iou = self.iou_metric.compute()
        self.log('val_dice', dice, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
        self.dice_metric.reset()
        self.iou_metric.reset()

    # def configure_optimizers(self):
    #     return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=5,  # self.t_max,
            eta_min=1e-6,  # self.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "cosine_annealing"
            }
        }