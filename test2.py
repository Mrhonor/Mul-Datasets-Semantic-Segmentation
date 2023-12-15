import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GumbelSoftmax(object):

    def __init__(self, probabilities, temperature):
        self.probs = probabilities
        self.temp = temperature

    def sample_gumbel(self, shape, eps=1e-20):
        """
        Sample from a Gumbel(0, 1)
        :param shape:
        :param eps:
        :return:
        """
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self):
        """
        Draw a sample from the Gumbel-Softmax distribution
        :return:
        """
        y = self.probs + self.sample_gumbel(self.probs.shape)
        return F.softmax(y / self.temp, dim=-1)

    def __call__(self, hard, is_training=True):
        """
        Sample from the Gumbel-Softmax
        If hard=True
            a) Forward pass:    argmax is taken
            b) Backward pass:   Gumbel-Softmax approximation is used
                                since it is differentiable
        else
            a) Gumbel-softmax used
        :param hard:
        :return:
        """
        y_sample = self.gumbel_softmax_sample()
        if hard or not is_training:
            # argmax of y sample
            y_hard = (y_sample == y_sample.max(dim=-1, keepdim=True)[0]).float()
            # let y_sample = y_hard but only allowing y_sample to be used for derivative
            y_sample = (y_hard - y_sample).detach() + y_sample
        return y_sample


class CategoricalVariableInitByP(object):
    """
    Initialise Categorical probabilities for each kernel by (p1, ps, p2)
    """

    def __init__(self, probs, num_kernels):
        """
        Type of initialisation for kernel probabilities
        :param probs: (p1, ps, p2)
        :parma num_kernels number of kernels
        """
        self.probs = probs
        self.num_kernel = num_kernels

    def __call__(self):
        """
        Create and initialise variables
        :return:
        """
        dirichlet_init_user = np.float32(np.asarray(self.probs))
        dirichlet_init = dirichlet_init_user * np.ones((self.num_kernel, 3), dtype=np.float32)
        return dirichlet_init
    

cat_probs_init = torch.tensor([1, 0, 0])

# cat_probs_init = np.float32(np.asarray(np.log(np.exp(cat_probs_init) - 1.0)))
# P_initialiser = CategoricalVariableInitByP(cat_probs_init, 2)
# dirichlet_init = torch.tensor(P_initialiser())
# this_dirichlet_p = F.softplus(dirichlet_init)
# this_dirichlet_p = this_dirichlet_p / torch.sum(this_dirichlet_p, dim=1, keepdim=True)

# Run if sampling from a categorical
# Can be if learning p (learn_cat=True) or constant p (learn_cat=False)
# Create object for categorical
# print(this_dirichlet_p)
cat_dist = GumbelSoftmax(cat_probs_init, 1e-20)
# Sample from mask - [3 by N] either one-hot (use_hardcat=True) or soft (use_hardcat=False)
cat_mask = cat_dist(hard=False, is_training=True)
for i in range(100):
    cat_mask += cat_dist(hard=False, is_training=True)   
print(cat_mask)

# cat_mask_unstacked = torch.unbind(cat_mask, dim=1).T()
# N = len(cat_mask_unstacked)
# cat_mask[:N-1] += N