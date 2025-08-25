#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import torch
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributions as distributions
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import convnext_base, ConvNeXt_Base_Weights



# In[ ]:


def get_resnet50(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.resnet50_ft = models.resnet50(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.resnet50_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_vgg16(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.vgg16_ft = models.vgg16(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.vgg16_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_resnext50(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.resnext50_32x4d_ft = models.resnext50_32x4d(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.resnext50_32x4d_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_alexnet(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.alexnet_ft = models.alexnet(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.alexnet_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_efficientnet_b0(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.efficientnet_b0_ft = models.efficientnet_b0(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.efficientnet_b0_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf

def get_convnext_base(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.convnext_base_ft = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.convnext_base_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf

class ProtoNetEncoder(nn.Module):
    def __init__(self, backbone='convnext_base', pretrained=True, proj_dim=None):
        super().__init__()

        # Load ConvNeXt backbone
        if backbone == 'convnext_base':
            weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            model = convnext_base(weights=weights)
            self.feature_dim = model.classifier[2].in_features  # usually 1024

            # Remove classification head
            self.encoder = nn.Sequential(
                model.features,  # ConvNeXt feature extractor
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        else:
            raise NotImplementedError(f"Backbone '{backbone}' is not supported yet.")

        # Optional projection head (e.g., to reduce dim)
        if proj_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            )
            self.feature_dim = proj_dim
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)         # shape: [B, feat_dim]
        x = self.projection(x)      # optional projection
        return x


# In[ ]:


def get_pretrain_model(name, output_shape=8):
    """Get the pre-trained model based on the name provided.
    Args:
        name (str): Name of the pre-trained model.
        output_shape (int): Number of output classes for the final layer.
    Returns:
        nn.Module: The pre-trained model with the final layer adjusted for the specified output shape.
    """
    if name == 'resnet50':
        return get_resnet50(output_shape)
    elif name == 'vgg16':
        return get_vgg16(output_shape)
    elif name == 'resnext50':
        return get_resnext50(output_shape)
    elif name == 'alexnet':
        return get_alexnet(output_shape)
    elif name == 'efficientnet_b0':
        return get_efficientnet_b0(output_shape)
    elif name == 'convnext_base':
        return get_convnext_base(output_shape)
    elif name == 'protonet_convnext_base':
        return ProtoNetEncoder(backbone='convnext_base', pretrained=True, proj_dim=None)
    else:
        return None


# In[ ]:


# print(get_pretrain_model('efficientnet_b0'))


# In[ ]:




