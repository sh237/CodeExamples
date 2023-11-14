
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output

class ImageTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes):
        super(ImageTransformer, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2)
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

def main():
    # Hyperparameters for the model, to be adjusted as needed.
    input_dim = (224, 224)  # Typically the spatial dimensions of the input image.
    d_model = 512  # Number of expected features in the input (required for the self-attention mechanism).
    nhead = 8  # Number of attention heads.
    num_encoder_layers = 6  # Number of sub-encoder-layers in the encoder.
    dim_feedforward = 2048  # Dimension of the feedforward network model.
    num_classes = 10  # Number of classes for classification.

    # Create a random image tensor to simulate one batch of image data.
    img_batch = torch.rand((8, 3, *input_dim))  # Simulated batch with batch_size = 8

    # Initialize the model.
    model = ImageTransformer(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes)

    # Forward pass of the model (without training just to demonstrate the functionality).
    out = model(img_batch)
    print(out.shape)  # Expected output: torch.Size([8, 10]) for the 10-class classification.

if __name__ == "__main__":
    main()
