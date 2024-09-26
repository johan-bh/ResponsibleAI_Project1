
import torch
import torch.nn as nn
# import torch.optim as optim
from torchvision import models
# from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.metrics import roc_auc_score
# from tqdm import tqdm
from torchmetrics import Accuracy,AUROC
from torchmetrics.classification import MultilabelAUROC
# import torch.nn.functional as F



class ResNet(nn.Module):
    def __init__(self, num_classes, lr, pretrained=True, model_scale='18', in_channel=1,
                 pos_weight=torch.tensor([20.0])):
        super(ResNet, self).__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_scale = model_scale
        self.in_channel = in_channel
        self.pos_weight = pos_weight

        # Load the appropriate ResNet model
        if self.model_scale == '18':
            self.model = models.resnet18(pretrained=self.pretrained)
        elif self.model_scale == '34':
            self.model = models.resnet34(pretrained=self.pretrained)
        elif self.model_scale == '50':
            self.model = models.resnet50(pretrained=self.pretrained)

        if self.in_channel == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if self.num_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight) 
            # self.loss_fn = nn.CrossEntropyLoss()
        elif self.num_classes > 1:
            self.loss_fn = nn.CrossEntropyLoss()

        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        # Define accuracy and AUROC based on the number of classes
        if self.num_classes == 1:
            self.accu_func = Accuracy(task="binary")
            self.auroc_func = AUROC(task='binary', average='macro')
        elif self.num_classes > 1:
            self.accu_func = Accuracy(task="multilabel", num_labels=num_classes)
            self.auroc_func = MultilabelAUROC(num_labels=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params_to_update = [param for param in self.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def process_batch(self, batch):
        img, lab = batch['image'], batch['label']
        device = lab.device
        # lab = lab.type(torch.LongTensor).to(device)

        out = self.forward(img)
        prob = torch.sigmoid(out) if self.num_classes == 1 else torch.softmax(out, dim=1)
        if self.num_classes == 2:
            prob = prob[:, 1] # Only need the probability of the positive class
        prob = prob.view(lab.shape)

        loss = self.loss_fn(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        # multi_auroc = self.auroc_func(prob, lab.long()) # noted that if there's only positive or negative cases, auroc means nothing
        return  prob, loss, multi_accu, None
    
    def predict(self, X):
        X_tensor = X.type(torch.FloatTensor) 
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            prob = torch.sigmoid(outputs) if self.num_classes == 1 else torch.softmax(outputs, dim=1)
            predicted = (prob > 0.5).float()
        return predicted.cpu().detach().numpy()
    
    def predict_proba(self, X):
        X_tensor = X.type(torch.FloatTensor) 
        self.model.eval()
        with torch.no_grad():
            if self.num_classes == 1:
                outputs = torch.sigmoid(self.model(X_tensor))
                # fairlearn need a output of 2 dimension
                outputs = torch.cat((1-outputs,outputs),dim=1)
            else: 
                outputs = torch.softmax(self.model(X_tensor), dim=1)
        return outputs.cpu().detach().numpy()