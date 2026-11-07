import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenEncoder(nn.Module):
    """Pretrained image encoder used as a fixed feature extractor."""
    def __init__(self, net, feat_dim, mean, std):
        super(FrozenEncoder, self).__init__()
        self.net = net
        self.feat_dim = feat_dim
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.net.requires_grad_(False)
        self.net.eval()

    def train(self, mode=True):
        # Frozen: keep dropout / norm layers in inference mode even when the head trains
        return super(FrozenEncoder, self).train(False)

    def forward(self, x):
        return self.net(x)


def _open_clip(model_name, pretrained=None):
    import open_clip  # imported lazily so tools that never build an encoder do not need it
    model = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)[0]
    visual = model.visual  # only the image tower is used; the text tower is dropped
    mean = getattr(visual, 'image_mean', None) or open_clip.OPENAI_DATASET_MEAN
    std = getattr(visual, 'image_std', None) or open_clip.OPENAI_DATASET_STD
    return FrozenEncoder(visual, visual.output_dim, mean, std)


def _dinov2(model_name):
    net = torch.hub.load('facebookresearch/dinov2', model_name)
    return FrozenEncoder(net, net.embed_dim, IMAGENET_MEAN, IMAGENET_STD)


BACKBONES = {
    # BioCLIP 2: ViT-L/14 trained on TreeOfLife-200M (biology / species images), 768-d
    'bioclip2': lambda: _open_clip('hf-hub:imageomics/bioclip-2'),
    # DINOv2 ViT-B/14 (plan default "DINOv2": leakage control, no species labels), 768-d
    'dinov2_b14': lambda: _dinov2('dinov2_vitb14'),
    # DINOv2 ViT-L/14, 1024-d; optional in A10 only
    'dinov2_l14': lambda: _dinov2('dinov2_vitl14'),
    # Original LTIC-Herb encoder: open_clip ViT-B/32 trained on LAION-2B, 512-d
    'clip_b32': lambda: _open_clip('ViT-B-32', pretrained='laion2b_s34b_b79k'),
    # CLIP ViT-B/16 with OpenAI weights: LIFT's backbone, for the like-for-like A11 comparison.
    # Projected embedding is 512-d (the 768-d width is pre-projection).
    'clip_b16': lambda: _open_clip('ViT-B-16', pretrained='openai'),
}


def _debug_tiny():
    # Random, untrained 32-d encoder for unit tests and smoke runs only (no download)
    torch.manual_seed(0)
    net = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48, 32))
    return FrozenEncoder(net, 32, IMAGENET_MEAN, IMAGENET_STD)


BACKBONES['debug_tiny'] = _debug_tiny


def build_backbone(name):
    if name not in BACKBONES:
        raise ValueError("unknown backbone '{}', choose from {}".format(name, sorted(BACKBONES)))
    return BACKBONES[name]()
