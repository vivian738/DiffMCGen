import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

import utils
from diffusion import diffusion_utils
from models.layers import RMSnormNoscale, Xtoy, Etoy, _cross_head_proj, masked_softmax, RMSNorm


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 norm_eps: float = 1e-6, layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = DCMHAttention(dx, de, dy, n_head, **kw)
        # self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.silu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class DCMHAttention(nn.Module):
    """ Self attention layer that also updates the representations on the edges. 
        DCMHA can be used in here"""
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = dx // n_head  # head_dim
        self.n_head = n_head
        
        # DCMHA
        self.query_input_dim = dx
        self.num_groups = 1
        self.num_heads_per_group = self.n_head // self.num_groups
        self.dynamic_w_hidden_dim = self.n_head*4
        self.dynamic_hidden_dim =  self.num_heads_per_group // (self.n_head//2)
        self.dw1_norm = RMSnormNoscale(dim=-1)   #stable training
        self.dw_m = nn.parameter.Parameter(torch.cat([
            torch.zeros(self.query_input_dim, self.num_groups, 4, self.dynamic_w_hidden_dim, dtype=torch.float32).reshape(self.query_input_dim, -1), 
            torch.zeros(self.query_input_dim, self.num_groups, self.num_heads_per_group * 4, dtype=torch.float32).squeeze(1)], dim=-1)) # E,(4*K + K)  K=2*N*I
        self.qkw_m = nn.parameter.Parameter(torch.zeros(self.num_groups, 4, self.dynamic_w_hidden_dim, self.dynamic_hidden_dim*2, self.num_heads_per_group).reshape(4,self.dynamic_w_hidden_dim,-1))

        # Attention
        self.wqkv = Linear(dx, 3 * dx, bias=False)
        
        self.node_gate = nn.Linear(dy, dx)
        self.edge_gate = nn.Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q, K, V = self.wqkv(X).split([self.dx, self.dx, self.dx], dim=-1)    # bs, n, dx    

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = (Q * x_mask).contiguous().view(Q.size(0), 1, Q.size(1), self.n_head, self.df)         # (bs, 1, n, n_head, df)      
        K = (K * x_mask).contiguous().view(K.size(0), K.size(1), 1, self.n_head, self.df)         # (bs, n, 1, n head, df)            # (bs, n, dx)
        V = (V * x_mask).contiguous().view(V.size(0), 1, V.size(1), self.n_head, self.df)         # (bs, 1, n, n_head, df)                          
        # diffusion_utils.assert_correctly_masked(Q, x_mask)
        
        # DCMHA to get Y (Compose(A, Q, K, theta-pre) before softmax)
        dw_hidden, dd = (X @ self.dw_m).split([2*2*self.n_head*(2*self.dynamic_hidden_dim), 2*2*self.n_head*1], -1)  # (bs, n, df)
        dw_hidden = F.gelu(dw_hidden) 
        dw_hidden = dw_hidden.view(dw_hidden.shape[:2]+(4,-1)) #bs n (4 df) -> bs n 4 df  # reshape
        dw = torch.einsum('B T C K, C K D -> B T C D', dw_hidden, self.qkw_m) # BT4K,4K(MI)->BT4(MI) (bs n 4 df)
        shape = (bs,n,2*2,-1,self.n_head)# if project_logits else (bs,n,2,n_head,-1)  # BT(pre/post)(q/k)IN
        w1, w2 = dw.view(shape).split(self.dynamic_hidden_dim,-2)
        w1 = self.dw1_norm(w1) # bs, n, 2*2, I, n_head
        pre_qw1, pre_kw1, post_qw1, post_kw1 = w1.unbind(2)  # BT(2{*2})IN->[BTIN]*4    (bs n 2 n_head)
        pre_qw2, pre_kw2, post_qw2, post_kw2 = w2.unbind(2)
        qkdd = F.tanh(dd).squeeze(-1).view(shape[:-2] + (self.n_head,)) # BT(2{*2})N1->BT(2{*2})N (bs n 4 n_head)
        pre_qdd, pre_kdd, post_qdd, post_kdd = qkdd.unbind(2)  # BT(2{*2})N->[BTN]*4   (bs n n_head)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Q.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        logits = _cross_head_proj(Y, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)    # (bs, n, n, n_head, df)
        del pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd
        torch.cuda.empty_cache()

        # Incorporate edge features to the self attention scores.

        # Incorporate y to E
        newE = logits.flatten(start_dim=3)                     # bs, n, n, dx
        newE = (torch.sigmoid(self.edge_gate(y)).unsqueeze(1).unsqueeze(1) + 1) * newE      # bs, n, n, dx

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, n, n, n_head
        # attn = masked_softmax(Y, softmax_mask, dim=2)
        probs = masked_softmax(logits, softmax_mask, dim=2)  # bs, n, n, n_head, df
        probs = _cross_head_proj(probs, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
        del post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd
        torch.cuda.empty_cache()

        # Compute weighted values
        # weighted_V = probs @ V  
        weighted_V = probs * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        dynamic_bias = torch.mean(weighted_V, dim=-1, keepdim=True)
        newX = (torch.sigmoid(self.node_gate(y)).unsqueeze(1) + dynamic_bias) * weighted_V
        
        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X and E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y
    


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)