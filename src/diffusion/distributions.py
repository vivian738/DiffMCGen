import torch
import copy

class DistributionNodes:
    def __init__(self, histogram):
        """ Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
            historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
            # prob = prob[prob !=0]
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-10)
        return log_p

import torch
from torch.distributions.categorical import Categorical
from tqdm import tqdm


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties.keys()   # qm9: gap [1]

        # iterate dataset, get data nodes and corresponding properties
        num_atoms = []
        prop_values = []
        prop_id = torch.tensor(list(properties.values()))
        for idx in tqdm(list(dataloader.dataset.indices())):
            data = dataloader.dataset.get(idx)
            tars = []
            # for prop_id in prop_ids:
            #     if prop_id == 11:
            #         tars.append(dataloader.dataset.sub_Cv_thermo(data).reshape(1))
            #     else:
            tars.append(data.y[0][prop_id])
            tars = torch.cat(tars)
            num_atoms.append(copy.deepcopy(data.rdmol).GetNumAtoms())
            prop_values.append(tars)
        num_atoms = torch.tensor(num_atoms)  # [N]
        prop_values = torch.stack(prop_values)  # [N, num_prop]
        # for i, prop in enumerate(self.properties):
        self.distributions[self.properties] = {}
        self._create_prob_dist(num_atoms, prop_values, self.distributions[self.properties])
        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(0, int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins #min(self.num_bins, values.size(-1))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            idx = int((val.mean() - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if idx == n_bins:
                idx = n_bins - 1
            histogram[idx] += 1
            # iterate all props
        probs = histogram / torch.sum(histogram)
        probs = Categorical(probs.clone().detach())
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        # print(self.normalizer)
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        # for prop in self.properties:
        dist = self.distributions[self.properties][n_nodes]
        idx = dist['probs'].sample((1,))
        val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
        val = self.normalize_tensor(val, self.properties)
        vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch_(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val
