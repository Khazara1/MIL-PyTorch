import torch
from torchmil.nn import masked_softmax
from torchvision.models import resnet18, ResNet18_Weights


class AttentionMILModel(torch.nn.Module):
    def __init__(self, output_dim, att_dim, dropout_rate):
        super().__init__()

        # Feature extractor
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        emb_dim = self.resnet.fc.in_features

        self.resnet.fc = torch.nn.Identity()


        self.fc1 = torch.nn.Linear(emb_dim, att_dim)
        self.fc2 = torch.nn.Linear(emb_dim, att_dim)
        self.fc3 = torch.nn.Linear(att_dim, 1)

        self.classifier = torch.nn.Linear(emb_dim, output_dim)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, X, mask, bag_size, return_att=False):
        batch_size = int(X.shape[0] / bag_size)

        # Process only instances that are not masked (i.e., valid instances, not padding)
        X = self.resnet(X[mask != 0])  # (batch_size * bag_size, emb_dim)

        # Put back the processed instances to their original positions, so that the shape is preserved (as if all instances, including padding, were processed)
        resnet_output = torch.zeros((batch_size * bag_size, X.shape[1]), device=X.device, dtype=X.dtype)
        resnet_output[mask != 0] = X
        X = resnet_output

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
        

class YourModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)  # Binary classification

    def forward(self, x):
        # Define the forward pass of your model here
        return self.model(x)