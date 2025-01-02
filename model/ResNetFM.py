import torch
import torch.nn as nn
import timm

class ResNetFM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        model_name = 'resnet50'
        self.resnet = timm.create_model(model_name, pretrained=True, num_classes=0)
        # self.context_processor = timm.create_model(model_name, pretrained=True, num_classes = 0)
        # self.main_enc = enc
        self.fm_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        # self.kd_projector = nn.Linear(2048, 1536)
        self.classifier = nn.Linear(2048 + 1536 , num_classes)
        
        self.fm_encoder.eval()  
        
    def forward(self, x):
        x_resnet = self.resnet(x)
        with torch.no_grad():
            fm_features = self.fm_encoder(x)
        
        x = torch.cat([x_resnet, fm_features], dim=1)
        x = self.classifier(x)
        return x 
    
    def infer(self, x):
        with torch.no_grad():
            x_resnet = self.resnet(x)
            fm_features = self.fm_encoder(x)
            x = torch.cat([x_resnet, fm_features], dim=1)
            x = self.classifier(x)
        return x 
        