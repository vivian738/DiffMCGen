import math
import torch
from models.egnn import EGNNSparseNetwork
from diffusion.edge import (_extend_graph_order, get_edge_encoder, MultiLayerPerceptron, _extend_to_radius_graph,
                                assemble_atom_pair_feature)
from diffusion.diffusion_utils import get_timestep_embedding, get_beta_schedule
from torch_scatter import scatter_mean
from diffusion.layers import eq_transform, get_distance, get_angles, get_torsions

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.type_as(x)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DualEdgeEGNN(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.edge_encoder_global = get_edge_encoder(cfg)
        # self.edge_encoder_local = get_edge_encoder(cfg)
        self.training = cfg.model.train
        self.context = cfg.model.context
        self.atom_type_input_dim = cfg.model.num_atom #contains simple tmb or charge or not qm9:5+1(charge) geom: 16+1(charge csd: 14+1
        self.num_convs = cfg.model.num_convs
        # self.num_convs_local = cfg.model.num_convs_local
        self.hidden_dim = cfg.model.hidden_dim
        self.atom_out_dim = cfg.model.num_atom  # contains charge or not
        self.soft_edge = cfg.model.soft_edge
        self.norm_coors = cfg.model.norm_coors
        self.time_emb = True
        self.cutoff=cfg.model.cutoff
        self.num_diffusion_timesteps = cfg.model.num_diffusion_timesteps
        # self.vae_context = cfg.model.vae_context if 'vae_context' in cfg.model else False

        '''
                timestep embedding
                '''
        if self.time_emb:
            self.temb = torch.nn.Module()
            self.temb.dense = torch.nn.ModuleList([
                torch.nn.Linear(self.hidden_dim,
                                self.hidden_dim * 4),
                torch.nn.Linear(self.hidden_dim * 4,
                                self.hidden_dim * 4),
            ])
            self.temb_proj = torch.nn.Linear(self.hidden_dim * 4,
                                             self.hidden_dim)


        if self.context is not None:
            self.atom_type_input_dim = self.atom_type_input_dim + 1

        self.encoder_global = EGNNSparseNetwork(
            n_layers=self.num_convs,
            feats_input_dim=self.atom_type_input_dim,
            feats_dim=self.hidden_dim,
            edge_attr_dim=self.hidden_dim,
            m_dim=self.hidden_dim,
            soft_edge=self.soft_edge,
            norm_coors=self.norm_coors,
            cutoff=self.cutoff
        )

        # self.encoder_local = EGNNSparseNetwork(
        #     n_layers=self.num_convs_local,
        #     feats_input_dim=self.atom_type_input_dim,
        #     feats_dim=self.hidden_dim,
        #     edge_attr_dim=self.hidden_dim,
        #     m_dim=self.hidden_dim,
        #     soft_edge=self.soft_edge,
        #     norm_coors=self.norm_coors,
        #     cutoff=self.cutoff
        # )
        
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * cfg.model.hidden_dim,
            [cfg.model.hidden_dim, cfg.model.hidden_dim // 2, 1],
            activation=cfg.model.mlp_act
        )
        # self.grad_local_dist_mlp = MultiLayerPerceptron(
        #     2 * cfg.model.hidden_dim,
        #     [cfg.model.hidden_dim, cfg.model.hidden_dim // 2, 1], 
        #     activation=cfg.model.mlp_act
        # )

        betas = get_beta_schedule(
            beta_schedule=cfg.model.beta_schedule,
            beta_start=cfg.model.beta_start,
            beta_end=cfg.model.beta_end,
            num_diffusion_timesteps=cfg.model.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = torch.nn.Parameter(betas, requires_grad=False)
        # # variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = torch.nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

    def forward(self, atom_type, pos, bond_index, bond_type, batch, time_step,
                edge_type=None, edge_index=None, edge_length=None, context=None):

        if len(self.context) > 0 and self.context is not None:
            atom_type = torch.cat([atom_type, context], dim=1)

        if self.time_emb:
            nonlinearity = torch.nn.ReLU()
            temb = get_timestep_embedding(time_step, self.hidden_dim)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
            temb = self.temb_proj(nonlinearity(temb))  # (G, dim)
            atom_type = torch.cat([atom_type, temb.index_select(0, batch)], dim=1)

        if edge_index is None or edge_type is None or edge_length is None:
            # bond_type = torch.ones(bond_index.size(1), dtype=torch.long).to(bond_index.device)
            # N = atom_type.size(0)
            # edge_index, edge_type = _extend_graph_order(
            #                         num_nodes=N, 
            #                         edge_index=bond_index, 
            #                         edge_type=bond_type, order=3
            # )
            edge_index, edge_type = _extend_to_radius_graph(
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                cutoff=self.cutoff,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1) # (E, 1)
        # local_edge_mask = is_radius_edge(edge_type) # (E, )
        # sigma_edge = torch.ones(size=(edge_index.size(1), 1), device=pos.device)  # (E, 1)
        
        # Emb time_step for edge
        if self.time_emb:
            node2graph = batch
            edge2graph = node2graph.index_select(0, edge_index[0])
            temb_edge = temb.index_select(0, edge2graph)

        # angle_index, angles = get_angles(mol_list)
        # angle_feature = torch.cat([angle_index, angles], dim=-1)
        # Encoding global
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type,
            edge_index = edge_index
        )
        if self.time_emb:
            edge_attr_global += temb_edge
        # EGNN
        node_attr_global, _ = self.encoder_global(
            z=atom_type,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
            batch=batch
        )
        """
        Assemble pairwise features
        """
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )  # (E_global, 2H)
        """
        Invariant features of edges (radius graph, global)
        """
        edge_inv_global = self.grad_global_dist_mlp(h_pair_global)
        # Encoding local
        # edge_attr_local = self.edge_encoder_local(
        #     edge_length=edge_length,
        #     edge_type=edge_type
        # )   # Embed edges
        # edge_attr_local += temb_edge

        # Local
        # node_attr_local,_ = self.encoder_local(
        #     z=atom_type,
        #     pos=pos,
        #     edge_index=edge_index[:, local_edge_mask],
        #     edge_attr=edge_attr_local[local_edge_mask],
        #     batch=batch
        # )
        # ## Assemble pairwise features
        # h_pair_local = assemble_atom_pair_feature(
        #     node_attr=node_attr_local,
        #     edge_index=edge_index[:, local_edge_mask],
        #     edge_attr=edge_attr_local[local_edge_mask],
        # )    # (E_local, 2H)
        ## Invariant features of edges (bond graph, local)
        # if isinstance(sigma_edge, torch.Tensor):
        #     edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge[local_edge_mask]) # (E_local, 1)
        # else:
        #     edge_inv_local = self.grad_local_dist_mlp(h_pair_local) * (1.0 / sigma_edge) # (E_local, 1)
        # return edge_inv_global, edge_inv_local, edge_index, edge_type, edge_length, local_edge_mask
        return edge_inv_global, edge_index, edge_type, edge_length

    def egn_process(self, pos, batch):

        # N = atom_type.size(0)
        node2graph = batch
        num_graphs = node2graph[-1] + 1

        # # Sample time step
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos.device)
        
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        a = self.alphas.index_select(0, time_step.long())  # (G, )

        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)

        """
        Independently
        - Perturb pos
        """
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + center_pos(pos_noise, batch) * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        # pos_perturbed = torch.nan_to_num(pos_perturbed, nan=0, posinf=1e9, neginf=-1e9)
        # pos_perturbed = a_pos.sqrt()*pos+(1.0 - a_pos).sqrt()*center_pos(pos_noise,batch)

        return pos_perturbed, a, node2graph, time_step

    def sampled_dynamics_pos_atom(self, edge_inv_global, edge_index, edge_length, 
                                  pos, batch, t, sigmas, i, j, w_global=1.0, local_start_sigma=1.0, global_start_sigma=0.5): # edge_inv_local, local_edge_mask, 
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a
        # sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        if sigmas[i] < local_start_sigma:
            node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask],
                                            edge_length[local_edge_mask])
        else:
            node_eq_local = 0
            node_score_local = 0

        # Global
        if sigmas[i] < global_start_sigma:
            edge_inv_global = edge_inv_global * (1 - local_edge_mask.view(-1, 1).float())
            node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
            node_eq_global = clip_norm(node_eq_global, limit=1000)
        else:
            node_eq_global = 0
            node_score_global = 0

            # Sum
        eps_pos = node_eq_global * w_global # + eps_pos_reg * w_reg +node_eq_local 

        # Update

        noise = center_pos(torch.randn_like(pos) + torch.randn_like(pos), batch)
        b = self.betas
        t = t[0]
        next_t = (torch.ones(1) * j).to(pos.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())

        atm1 = at_next
        beta_t = 1 - at / atm1
        e = -eps_pos
        mean = (pos - beta_t * e) / (1 - beta_t).sqrt()
        mask = 1 - (t == 0).float()
        logvar = beta_t.log()
        pos_next = mean + mask * torch.exp(
            0.5 * logvar) * noise  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

        pos = pos_next
        # atom_type = atom_next
        pos = center_pos(pos, batch)
        # atom_f = atom_type[:, :-1] * 4
        # atom_charge =  torch.round(atom_type[:, -1:] * 10).long()
        return pos


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
