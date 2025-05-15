import pytorch_lightning as pl
import torch
from transformers import ViTMAEConfig, ViTMAEForPreTraining


class MAEPretrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        # Build config from scratch
        self.model = get_mae_model(cfg)  # Instantiate MAE without pretrained weights

    def forward(self, batch):
        # Forward pass through the model
        pixels = batch["pixel_values"]
        outputs = self.model(pixel_values=pixels)
        return outputs

    def training_step(self, batch, batch_idx):
        pixels = batch["pixel_values"]    # (B, 1, H, W)
        outputs = self.model(pixel_values=pixels)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 1) Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg['lr'], weight_decay=0.005)

        # 2) Scheduler: Cosine Annealing from lr -> eta_min over t_max epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg['t_max'],
            eta_min=self.cfg['eta_min']
        )

        # 3) Wrap in Lightning dict
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",      # call once per epoch
                "frequency": 1,
                "reduce_on_plateau": False,
                "name": "cosine_lr"
            }
        }
    

def get_mae_model(cfg, pretrained_weights:str=None):
    # Build config from scratch, then force 1 input channel
    config = ViTMAEConfig(
        image_size=cfg['model']['image_size'],
        patch_size=cfg['model']['patch_size'],
        encoder_hidden_size=cfg['model']['encoder_hidden_size'],
        decoder_hidden_size=cfg['model']['decoder_hidden_size'],
        mask_ratio=cfg['model']['mask_ratio']
    )
    config.num_channels = cfg['num_channels'] 
    # Instantiate MAE without pretrained weights
    model = ViTMAEForPreTraining(config)
    # Load pretrained weights if provided
    if pretrained_weights:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model