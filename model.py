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

import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB

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

class ClinicalBinnedOneHot(nn.Module):
    def __init__(self, cat_vocab_sizes: dict):
        super().__init__()

        self.age_bins = 6     # 1..6
        self.td_bins = 4      # 1..4

        self.eth_dim = int(cat_vocab_sizes["ETHNIC_GROUP_DESC"])
        self.race_dim = int(cat_vocab_sizes["race"])

        self.out_dim = self.age_bins + self.td_bins + self.eth_dim + self.race_dim

    def forward(self, clin_num: torch.Tensor, clin_cat: torch.Tensor):
        age_idx = (clin_num[:, 0] - 1).clamp(0, self.age_bins - 1)
        td_idx = (clin_num[:, 1] - 1).clamp(0, self.td_bins - 1)

        # cumulative / ordinal one-hot
        age_range = torch.arange(self.age_bins, device=clin_num.device).unsqueeze(0)
        age_oh = (age_range <= age_idx.unsqueeze(1)).float()

        td_range = torch.arange(self.td_bins, device=clin_num.device).unsqueeze(0)
        td_oh = (td_range <= td_idx.unsqueeze(1)).float()

        # standard one-hot
        eth_oh = F.one_hot(clin_cat[:, 0], num_classes=self.eth_dim).float()
        race_oh = F.one_hot(clin_cat[:, 1], num_classes=self.race_dim).float()

        return torch.cat([age_oh, td_oh, eth_oh, race_oh], dim=1)


class ClinicalOnlyClassifier(nn.Module):
    def __init__(
        self,
        cat_vocab_sizes: dict,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()
        if cat_vocab_sizes is None:
            raise ValueError("cat_vocab_sizes required")

        self.enc = ClinicalBinnedOneHot(cat_vocab_sizes)
        in_dim = self.enc.out_dim

        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        layers = []
        d = in_dim
        for _ in range(int(depth)):
            layers += [
                nn.Linear(d, int(hidden_dim)),
                act,
                nn.Dropout(p=float(dropout)),
            ]
            d = int(hidden_dim)

        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, inputs, return_features: bool = False):
        clin_num = inputs["clin_num"]
        clin_cat = inputs["clin_cat"]

        z = self.enc(clin_num, clin_cat)
        logits = self.net(z)

        if return_features:
            return logits, z
        return logits 




class ClinicalAgeDensityClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()

        self.td_bins = 4
        in_dim = 1 + self.td_bins   # age continuous + td cumulative one-hot

        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()

        layers = []
        d = in_dim
        for _ in range(int(depth)):
            layers += [
                nn.Linear(d, int(hidden_dim)),
                act,
                nn.Dropout(p=float(dropout)),
            ]
            d = int(hidden_dim)

        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, inputs, return_features: bool = False):
        age = inputs["age"]   # [B, 1]
        td = inputs["td"]     # [B]

        td_idx = (td - 1).clamp(0, self.td_bins - 1)

        td_range = torch.arange(self.td_bins, device=td.device).unsqueeze(0)
        td_cum_oh = (td_range <= td_idx.unsqueeze(1)).float()

        x = torch.cat([age, td_cum_oh], dim=1)
        logits = self.net(x)

        if return_features:
            return logits, x
        return logits


def build_logreg_classifier(params: dict, seed: int = 42):
    model = LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        random_state=seed,
    )
    return model


def build_xgb_classifier(params: dict, seed: int = 42, scale_pos_weight: float = 1.0):
    model = XGBClassifier(
        objective=params["objective"],
        eval_metric=params["eval_metric"],
        tree_method=params["tree_method"],
        random_state=seed,
        n_jobs=params["n_jobs"],
        scale_pos_weight=scale_pos_weight,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
    )
    return model

def build_dt_classifier(params: dict, seed: int = 42):
    model = DecisionTreeClassifier(
        criterion=params["criterion"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        random_state=seed,
    )
    return model


def build_rf_classifier(params: dict, seed: int = 42):
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        criterion=params["criterion"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=seed,
        n_jobs=1,
    )
    return model


def build_svm_classifier(params: dict, seed: int = 42):
    model = LinearSVC(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=params["max_iter"],
        dual=params["dual"],
        random_state=seed,
    )
    return model


def build_nb_classifier(params: dict):
    model = GaussianNB(
        var_smoothing=params["var_smoothing"],
    )
    return model

def build_rbf_svm_classifier(params: dict, seed: int = 42):
    model = SVC(
        C=params["C"],
        kernel=params["kernel"],
        gamma=params["gamma"],
        class_weight=params["class_weight"],
        probability=False,
        random_state=seed,
    )
    return model