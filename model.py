import torch
from torch import nn
import timm
from torchmil.nn import masked_softmax
from torchvision.models import (
    resnet18, resnet50,
    ResNet18_Weights, ResNet50_Weights,
    convnext_base,
    ConvNeXt_Base_Weights,
)
from timm.models import create_model

TAR_PATH = "models/convnext/best_convnext_fold_0.pth.tar"

def _pick_state_dict(ckpt: dict) -> dict:
    for k in ("state_dict_ema", "ema_state_dict", "state_dict", "model", "net"):
        sd = ckpt.get(k)
        if isinstance(sd, dict):
            return sd
    return ckpt if isinstance(ckpt, dict) else {}


def _strip_prefix(s: str, pref: str) -> str:
    return s[len(pref):] if s.startswith(pref) else s


def _load_tiny_ckpt_into_timm_convnext(model: nn.Module, ckpt_path: str) -> None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint nie jest dict.")

    sd_raw = _pick_state_dict(ckpt)
    if not isinstance(sd_raw, dict) or len(sd_raw) == 0:
        raise ValueError("Nie znaleziono state_dict w checkpoint.")

    sd = {_strip_prefix(k, "module."): v for k, v in sd_raw.items()}

    model_keys = set(model.state_dict().keys())
    model_has_backbone = any(k.startswith("backbone.") for k in model_keys)
    ckpt_has_backbone = any(k.startswith("backbone.") for k in sd.keys())
    if ckpt_has_backbone and (not model_has_backbone):
        sd = {_strip_prefix(k, "backbone."): v for k, v in sd.items()}

    drop_prefixes = ("head.", "fc.", "classifier.")
    sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}

    target = model.state_dict()
    loadable = {
        k: v for k, v in sd.items()
        if (k in target) and torch.is_tensor(v) and (tuple(v.shape) == tuple(target[k].shape))
    }
    model.load_state_dict(loadable, strict=False)


class AttentionMILModel(torch.nn.Module):
    def __init__(self, backbone="resnet18", output_dim=1, params={}, emb_dim=768):
        super().__init__()

        att_dim = params["att_dim"]
        dropout_rate = params["dropout_rate"]

        # Feature extractor
        if backbone == "resnet18":
            self.fe = resnet18(weights=ResNet18_Weights.DEFAULT)
            emb_dim = self.fe.fc.in_features
            self.fe.fc = torch.nn.Identity()
        elif backbone == "resnet50":
            self.fe = resnet50(weights=ResNet50_Weights.DEFAULT)
            emb_dim = self.fe.fc.in_features
            self.fe.fc = torch.nn.Identity()
        elif backbone == "convnext_tiny":
            self.fe = ConvNextFeatureExtractor()
            self.fe = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, global_pool="avg")
            _load_tiny_ckpt_into_timm_convnext(self.fe, TAR_PATH)
            emb_dim = self.fe.num_features

            # Freeze params to avoid OOM error
            for param in self.fe.parameters():
                param.requires_grad = False

        else:
            raise ValueError("Unsupported backbone provided")

        self.backbone = backbone

        self.fc1 = torch.nn.Linear(emb_dim, att_dim)
        self.fc2 = torch.nn.Linear(emb_dim, att_dim)
        self.fc3 = torch.nn.Linear(att_dim, 1)

        self.classifier = torch.nn.Linear(emb_dim, output_dim)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, X, mask, bag_size, return_att=False):
        batch_size = int(X.shape[0] / bag_size)

        # Process only instances that are not masked (i.e., valid instances, not padding)          
        X = self.fe(X[mask != 0])  # (batch_size * bag_size, emb_dim)

        # Put back the processed instances to their original positions, so that the shape is preserved (as if all instances, including padding, were processed)
        fe_output = torch.zeros((batch_size * bag_size, X.shape[1]), device=X.device, dtype=X.dtype)
        fe_output[mask != 0] = X
        X = fe_output

        # Reshaping to separate bags from batches
        X = X.reshape((batch_size, bag_size, -1))  # (batch_size, bag_size, emb_dim)
        mask = mask.reshape((batch_size, bag_size))  # (batch_size, bag_size)

        H = torch.tanh(self.fc1(X))  # (batch_size, bag_size, att_dim)
        att = torch.sigmoid(self.fc2(X))  # (batch_size, bag_size, att_dim)

        att = torch.mul(H, att) # (batch_size, bag_size, att_dim)
        att = self.fc3(att) # (batch_size, bag_size, 1)

        att_s = masked_softmax(att, mask)  # (batch_size, bag_size, 1)
        # att_s = torch.nn.functional.softmax(att, dim=1)
        X = torch.bmm(att_s.transpose(1, 2), X).squeeze(1)  # (batch_size, emb_dim)
        X = self.dropout(X)
        y = self.classifier(X).squeeze(1)  # (batch_size,)
        if return_att:
            return y, att_s
        else:
            return y
        

class ConvNextFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = create_model(
                'convnext_small.fb_in22k_ft_in1k_384',
                num_classes=1,
                in_chans=3,
                pretrained=False,
                checkpoint_path=TAR_PATH,
                global_pool='max',
            )
        
        emb_dim = self.feature_extractor.head.fc.in_features
        self.feature_extractor.head.fc = torch.nn.Identity()

    def forward(self, X):
        return self.feature_extractor(X)

class StandardImageModel(torch.nn.Module):
    def __init__(self, backbone="resnet18", num_classes=1, pretrained=True, params={}):
        super().__init__()
        backbone = backbone.lower()

        dropout_rate = params["dropout_rate"]

        if backbone == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            in_f = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=float(dropout_rate)),
                nn.Linear(in_f, num_classes),
            )

        elif backbone == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            in_f = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=float(dropout_rate)),
                nn.Linear(in_f, num_classes),
            )

        elif backbone == "convnext_tiny":
            bb = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, global_pool="avg")
            if pretrained:
                _load_tiny_ckpt_into_timm_convnext(bb, TAR_PATH)
            self.model = nn.Sequential(
                bb,
                nn.Dropout(p=float(dropout_rate)),
                nn.Linear(bb.num_features, num_classes),
            )

        elif backbone == "convnext_base":
            self.model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
            in_f = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(
                nn.Dropout(p=float(dropout_rate)),
                nn.Linear(in_f, num_classes),
            )

        else:
            print(backbone)
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        # Define the forward pass of your model here
        return self.model(x)