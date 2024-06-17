import torch
import torch.nn as nn

class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)

class RMSNorm(nn.Module):
    def __init__(self, hid_dim=128, p=0.0625, epsilon=1e-6, device=None, dtype=None):
        """Root Mean Square Layer Normalization

        Args:
            hid_dim (int, optional): model size. Defaults to 128.
            p (int, optional): partial RMSNorm. valid value [0,1], Defaults to -1.
            epsilon (_type_, optional): epsilonm value. Defaults to 1e-8.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.hid_dim = hid_dim
        self.epsilon = epsilon
        self.p = p
        self.scale = nn.parameter.Parameter(data=torch.ones(self.hid_dim, **factory_kwargs))

    def forward(self, inputs):
        if self.p < 0. or self.p > 1.:
            norm_x = inputs.norm(2, dim=-1, keepdim=True)
            d_x = self.hid_dim
        else:
            partial_size = int(self.hid_dim * self.p)
            partial_x, _ = torch.split(inputs, [partial_size, self.hid_dim - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        var = norm_x * d_x ** (-1. / 2)
        normed_inputs = inputs / (var + self.epsilon)
        normed_inputs = self.scale * normed_inputs
        return normed_inputs

class RMSnormNoscale(nn.Module):
    
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim 
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs
    
def unbind(ary, n, dim=0):
    return [torch.squeeze(a, dim=dim) for a in torch.split(ary, ary.shape[dim] // n, dim=dim)]

def _cross_head_proj(inputs, qw1, qw2, kw1, kw2, qdd, kdd):
    out = inputs 
    for i in range(2): # qw1.shape[-2]):
        qhidden = torch.einsum('BTSNK, BTN->BTSK', inputs, qw1[..., i, :])  
        qout = torch.einsum('BTSK, BTN->BTSNK', qhidden, qw2[..., i, :]) 
        out = out + qout
        khidden = torch.einsum('BTSNK, BSN->BTSK', inputs, kw1[..., i, :])  
        kout = torch.einsum('BTSK, BSN->BTSNK', khidden, kw2[..., i, :])  
        out = out + kout
    qdout = inputs * qdd.unsqueeze(2).unsqueeze(-1); out = out + qdout  # BTSNK,(BTN->BT1N1)->BNTS
    kdout = inputs * kdd.unsqueeze(1).unsqueeze(-1); out = out + kdout  # BTSNK,(BSN->B1SN1)->BNTS
    return out