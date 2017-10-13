import math
import numpy as np
import sampyl as smp

import torch
from torch.nn.parameter import Parameter
from HyperSphere.GP.kernels.modules.kernel import Kernel, log_lower_bnd


class Stationary(Kernel):

	def __init__(self, ndim, input_map=None, max_ls=None):
		super(Stationary, self).__init__(input_map)
		if max_ls is None:
			self.max_log_ls = np.log(2.0 * self.ndim ** 0.5)
		self.max_log_ls = np.log(max_ls)
		self.ndim = ndim
		self.log_ls = Parameter(torch.FloatTensor(ndim))

	def reset_parameters(self):
		super(Stationary, self).reset_parameters()
		self.log_ls.data.uniform_().mul_(self.max_ls).log_()

	def init_parameters(self, amp):
		super(Stationary, self).init_parameters(amp)
		self.log_ls.data.fill_(self.max_log_ls - np.log(2.0))

	def out_of_bounds(self, vec=None):
		if vec is None:
			if not super(Stationary, self).out_of_bounds():
				return (self.log_ls.data > self.max_log_ls).any() or (self.log_ls.data < log_lower_bnd).any()
		else:
			if not super(Stationary, self).out_of_bounds(vec[:super(Stationary, self).n_params()]):
				return (vec[1:] > self.max_log_ls).any() or (vec < log_lower_bnd).any()
		return True

	def n_params(self):
		cnt = super(Stationary, self).n_params() + self.ndim
		return cnt

	def param_to_vec(self):
		return torch.cat([super(Stationary, self).param_to_vec(), self.log_ls.data])

	def vec_to_param(self, vec):
		n_param_super = super(Stationary, self).n_params()
		super(Stationary, self).vec_to_param(vec[:n_param_super])
		self.log_ls.data = vec[n_param_super:]

	def elastic_vec_to_param(self, vec, func):
		n_param_super = super(Stationary, self).n_params()
		super(Stationary, self).vec_to_param(vec[:n_param_super])
		self.log_ls.data = func(vec[n_param_super:])

	def prior(self, vec):
		n_param_super = super(Stationary, self).n_params()
		return super(Stationary, self).prior(vec[:n_param_super]) + smp.uniform(np.exp(vec[n_param_super:]), lower=np.exp(log_lower_bnd), upper=np.exp(self.max_log_ls))

	def __repr__(self):
		return self.__class__.__name__ + ' (' + 'dim=' + str(self.ndim) + ')'
