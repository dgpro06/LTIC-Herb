import torch
import torch.nn as nn

from networks.backbones import build_backbone


class LTICNet(nn.Module):
    """LTIC head (as in CLIP_VIT_Large) on top of a frozen, swappable image encoder.

    C0 (plan §3.1): the shared classifier has no bias. The IJCNN code trained with self.fc(out)
    (with bias) but evaluated torch.mm(out, weight.T) (bias dropped), so test logits differed from
    training logits by a learned per-class offset. gamma is LTIC's test-time weight normalization
    w / ||w||^gamma; gamma = 0 makes eval logits identical to training logits.
    """
    def __init__(self, num_classes=1000, backbone='bioclip2', dropout=False, gamma=0.0):
        super(LTICNet, self).__init__()
        self.encoder = build_backbone(backbone)
        self.gamma = gamma

        self.intermediate = nn.Sequential(
            nn.Linear(self.encoder.feat_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 170)
        )

        # 170 // 3 = 56 per branch; the last 2 dims are unused (R2, harmless)
        self.fc = nn.Linear(56, num_classes, bias=False)
        self.dropout_mark = dropout
        self.dropout = nn.Dropout(p=0.5) if dropout else None

    def extract(self, images):
        # Mixed precision for the frozen encoder only; the head runs in fp32
        with torch.no_grad(), torch.autocast(images.device.type, enabled=images.is_cuda):
            return self.encoder(images).float()

    def forward(self, x):
        # 2-D input = features already extracted by the encoder (--cache_features / --feature_dir)
        x = x.float() if x.dim() == 2 else self.extract(x)
        batch_size = x.size(0)

        x = self.intermediate(x)

        # Split features into the three branches and share one classifier
        c = x.size(1) // 3
        x1, x2, x3 = x[:, :c], x[:, c:c*2], x[:, c*2:c*3]
        out = torch.cat((x1, x2, x3), dim=0)

        if self.dropout_mark:
            out = self.dropout(out)

        weight = self.fc.weight
        if not self.training and self.gamma:
            norm = torch.norm(weight, 2, 1, keepdim=True)
            weight = weight / torch.pow(norm, self.gamma)
        y = torch.mm(out, torch.t(weight))

        return y[:batch_size], y[batch_size:batch_size*2], y[batch_size*2:batch_size*3]
