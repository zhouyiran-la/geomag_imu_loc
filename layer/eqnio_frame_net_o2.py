import torch
import torch.nn.functional as F
from torch import nn

class VNLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
    ):
        super().__init__()

        self.vector_linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, vector):
        return self.vector_linear(vector)
    
class NonLinearity(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_in,
        scalar_dim_out,
    ):
        super().__init__()
        self.scalar_dim_out = scalar_dim_out
        self.dim_out = dim_out

        self.linear = nn.Linear(dim_in+scalar_dim_in, dim_out+scalar_dim_out, bias=False)
        self.layer_norm = LayerNorm(dim_out+scalar_dim_out)

        
    def forward(self, vector, scalar):
        x = torch.concatenate((torch.norm(vector, dim=-2), scalar), dim=-1)
        x = self.linear(x)
        x = nn.ReLU()(x)
        x = self.layer_norm(x)
        if self.scalar_dim_out == 0:
            return x[..., :self.dim_out].unsqueeze(-2) * (vector/torch.norm(vector, dim=-2).clamp(min = 1e-6).unsqueeze(-2))
        return x[..., :self.dim_out].unsqueeze(-2) * (vector/torch.norm(vector, dim=-2).clamp(min = 1e-6).unsqueeze(-2)), x[..., -self.scalar_dim_out:]
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6): # dim is the vector dimension (i.e., 2)
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim = -2)
        x = x / norms.clamp(min = self.eps).unsqueeze(-2)
        return x * self.ln(norms).unsqueeze(-2)
    
class MeanPooling_layer(nn.Module):
    def __init__(
        self,
        dim = 1
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, vector, scalar):
        return torch.mean(vector, dim=self.dim), torch.mean(scalar, dim=self.dim)
    
class Convolutional(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scalar_dim_out,
        scalar_dim_in,
        stride = 1,
        padding='same',
        kernel=(16,1),
        bias = False
    ):
        super().__init__()
        self.conv_layer_vec = nn.Conv2d(in_channels=dim_in, out_channels=dim_out, stride=stride, kernel_size=kernel, padding=padding, bias=bias, padding_mode='replicate')
        self.conv_layer_sca = nn.Conv2d(in_channels=scalar_dim_in, out_channels=scalar_dim_out, stride=stride, kernel_size=kernel, padding=padding, bias=bias, padding_mode='replicate')

        
    def forward(self, vector, scalar):
        return self.conv_layer_vec(vector.permute(0,3,1,2)).permute(0,2,3,1), self.conv_layer_sca(scalar.unsqueeze(-2).permute(0,3,1,2)).permute(0,2,3,1).squeeze(-2)

class EqNIOFrameNetO2(nn.Module):
    """Strictly replicated FrameNet from Eq_Motion_Model_o2 (up to frame estimation).
    It takes (vector, scalar) features as in EqNIO and outputs the 2x2 frame.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 scalar_dim_in,
                 pooling_dim,
                 hidden_dim,
                 scalar_hidden_dim,
                 depth,
                 stride=1,
                 padding='same',
                 kernel=(16,1),
                 bias=False):
        super().__init__()
        # --- copied from Eq_Motion_Model_o2, excluding ronin backbone ---
        self.vnlinear_layer0 = VNLinear(dim_in=dim_in, dim_out= hidden_dim)
        self.slinear_layer0 = nn.Linear(scalar_dim_in,scalar_hidden_dim, bias=False)
        self.nonlinearity0 = NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim,
                                          scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([ 
                Convolutional(dim_in=hidden_dim, dim_out= hidden_dim, scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim,
                              stride = stride, padding=padding, kernel = kernel, bias=bias),
                NonLinearity(dim_in=hidden_dim, dim_out= hidden_dim,
                             scalar_dim_in=scalar_hidden_dim, scalar_dim_out=scalar_hidden_dim),
                VNLayerNorm(hidden_dim), ##for vector
                LayerNorm(scalar_hidden_dim), ## for scalar
            ]))
        self.pooling_layer1 = MeanPooling_layer(dim = pooling_dim)

        ## MLP- lin (copied)
        self.linear_layer = VNLinear(dim_in=hidden_dim, dim_out=dim_out)
        self.linear_layer2 = VNLinear(dim_in=dim_out, dim_out=dim_out)

    def forward(self, vector, scalar):
        """vector: (B,T,2,dim_in), scalar: (B,T,scalar_dim_in)"""
        v = torch.clone(vector)
        s = torch.clone(scalar)
        v = self.vnlinear_layer0(v)
        s = self.slinear_layer0(s)
        v,s = self.nonlinearity0(v,s)
        ## conv blocks
        for conv, nl, vnln, sln in self.layers: # type: ignore
            v,s = conv(v,s)
            v,s = nl(v,s)
            v = vnln(v)
            s = sln(s)

        v,s = self.pooling_layer1(v,s)

        # --- frame estimation section copied verbatim from Eq_Motion_Model_o2 forward ---
        v = self.linear_layer(v)
        v = self.linear_layer2(v)

        # v: (B,2,dim_out) after pooling and linear; choose first two channels as axes
        v1 = v[...,0] / torch.norm(v[...,0], dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        # Gram-Schmidt
        v2 = v[...,1] - (v[...,1].unsqueeze(-2)@v1.unsqueeze(-1)).squeeze(-1)*v1
        v2 = v2/torch.norm(v2, dim=-1,keepdim=True).clamp(min = 1e-8) #(B,d)
        frame = torch.stack([v1,v2],dim=-1)

        frame = frame.permute(0,2,1)  # (B,2,2)
        return frame
