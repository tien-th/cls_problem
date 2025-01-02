import torch
import torch.nn as nn
import timm




class ResNetKdFM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        model_name = 'resnet50'
        enc = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.enc = enc
        
        self.kd_projector = nn.Linear(2048, 1536)
        self.classifier = nn.Linear(2048, num_classes)
        
        self.tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        
        self.tile_encoder.eval()    # tile for KD only
        
    def forward(self, x):
        x_main = self.enc(x)
        with torch.no_grad():
            x_fm = self.tile_encoder(x)
        logits = self.classifier(x_main)
        x_kd = self.kd_projector(x_main)
        outputs = (logits, x_kd, x_fm) 
        return outputs
    
    def infer(self, x):
        with torch.no_grad():
            x = self.enc(x)
            logits = self.classifier(x)
        return logits
    
        