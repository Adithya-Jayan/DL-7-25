
## Required Libraries and Packages
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None):
        # Flatten to 3D if needed
        original_Q_shape = Q.shape
        original_K_shape = K.shape
        
        if Q.dim() > 3:
            Q = Q.view(-1, Q.size(-2), Q.size(-1))
        if K.dim() > 3:
            K = K.view(-1, K.size(-2), K.size(-1))
            
        batch_size = Q.size(0)
        
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        
        # Properly reshape for multi-head attention
        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, dim_split).transpose(1, 2).contiguous().view(batch_size * self.num_heads, -1, dim_split)
        
        Q_ = split_heads(Q)
        K_ = split_heads(K) 
        V_ = split_heads(V)

        # Compute attention scores
        scores = Q_.bmm(K_.transpose(1,2)) / math.sqrt(dim_split)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask_expanded = mask.unsqueeze(1).repeat(1, self.num_heads, 1).view(batch_size * self.num_heads, 1, -1)
            # Set masked positions to large negative value
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        A = torch.softmax(scores, 2)
        O = Q_ + A.bmm(V_)
        
        # Reshape back
        O = O.view(batch_size, self.num_heads, -1, dim_split).transpose(1, 2).contiguous().view(batch_size, -1, self.dim_V)
        
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask=mask)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask=mask)

# Updated Set Transformer with masking support
class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc1 = ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)
        self.enc2 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.dec1 = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec2 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.dec3 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.final_layer = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, mask=None):
        # Encoder
        X = self.enc1(X, mask=mask)
        X = self.enc2(X, mask=mask)
        
        # Decoder
        X = self.dec1(X, mask=mask)
        X = self.dec2(X)  # PMA output doesn't need masking
        X = self.dec3(X)
        X = self.final_layer(X)
        
        return X
    
# Custom Dataset class to handle masks
class VariableSetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets, masks):
        self.encodings = torch.FloatTensor(encodings)
        self.targets = torch.FloatTensor(targets)
        self.masks = torch.FloatTensor(masks)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx], self.targets[idx], self.masks[idx]

# Create Dataloader class
class NewsDataset(Dataset):
    def __init__(self, encodings, price_changes):
        self.encodings = encodings
        self.price_changes = price_changes
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):

        # Multiply encodings by sentiment (broadcasting)
        input_data = self.encodings[idx]
        target = self.price_changes[idx]  # Single value, not array


        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)