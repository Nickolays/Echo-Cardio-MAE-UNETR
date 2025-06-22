import torch

from src.UNETR import UNETR


mae_cfg_path = "models/mae_pretrained/config.json"
mae_ckpt_path = "models/mae_pretrained.pth"  # or .pth

model = UNETR(
    mae_config_path=mae_cfg_path,
    mae_checkpoint_path=mae_ckpt_path,
    num_classes=2,
    freeze_encoder_blocks=0
)

x = torch.randn(2, 3, 224, 224)
seg_logits = model(x)  # shape (2, 4, 224, 224)

print(seg_logits.shape)  # should be (2, 4, 224, 224)