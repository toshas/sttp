import numpy as np
import torch

try:
    from torch_householder import torch_householder_orgqr as householder_product
except ImportError:
    from torch.linalg import householder_product


def dof_stiefel(rank, n):
    assert rank <= n
    return rank * n - (rank * (rank + 1) // 2)


def dof_stiefel_canonical(rank, n):
    assert rank <= n
    return rank * (n - rank)


class StiefelHouseholderCanonical(torch.nn.Module):
    has_penalty = False

    def __init__(
            self, batch, rank, n, is_thin, init_mode,
            init_std=0.0001, init_seed=None, init_qr_mul_sgn_diag_r=False,
            handle_det_sign_pos=False,
    ):
        super().__init__()
        assert 0 < rank <= n
        self.batch = batch
        self.rank = rank
        self.n = n
        self.is_thin = is_thin and (rank != n)
        self.num_param = batch * dof_stiefel_canonical(rank, n)
        rng = np.random if init_seed is None else np.random.RandomState(init_seed)
        self.param = torch.nn.Parameter(torch.zeros(batch, n-rank, rank))
        self.register_buffer('eye', torch.eye(n, rank).unsqueeze(0).repeat(batch, 1, 1))
        if handle_det_sign_pos and rank == n and n % 2 == 1:
            self.register_buffer('negate_last_col', torch.ones(1, 1, n))
            self.negate_last_col[-1] = -1
        with torch.no_grad():
            if init_mode == 'eye':
                self.param.zero_()
            elif init_mode in ('qr_eye_randn', 'qr_randn'):
                if init_mode == 'qr_eye_randn':
                    tmp = (
                        self.eye +
                        init_std * torch.from_numpy(rng.randn(batch, n, rank)).clamp(min=-1, max=1)
                    )
                else:
                    tmp = torch.from_numpy(rng.randn(batch, n, rank))
                param = []
                for m in tmp:
                    q, r = torch.linalg.qr(m)
                    if init_qr_mul_sgn_diag_r:
                        q *= r.diag().sign().view(1, rank)
                    hh = q.geqrf()[0].unsqueeze(0)
                    param.append(hh[:, rank:, :])
                param = torch.cat(param, dim=0)
                self.param.copy_(param)
            else:
                raise ValueError(f'Unknown orthogonal initialization {init_mode}')

    def forward(self):
        param = torch.cat((
            self.eye[:, :self.rank, :],
            self.param,
        ), dim=1)
        param_norm = 2 / (param * param).sum(dim=1)
        thin = householder_product(param, param_norm)
        if hasattr(self, 'negate_last_col'):
            thin = thin * self.negate_last_col
        if not self.is_thin:
            return thin.permute(0, 2, 1)
        return thin

    def dof(self):
        return self.num_param

    def flops(self):
        return self.batch * self.rank * (4 * self.n * self.rank + 3 * self.n + 1)


class StiefelHouseholder(torch.nn.Module):
    has_penalty = False

    def __init__(
            self, batch, rank, n, is_thin, init_mode,
            init_std=0.0001, init_seed=None, init_qr_mul_sgn_diag_r=False,
            handle_det_sign_pos=False,
    ):
        super().__init__()
        assert 0 < rank <= n
        self.batch = batch
        self.rank = rank
        self.n = n
        self.is_thin = is_thin and (rank != n)
        self.num_param = batch * dof_stiefel(rank, n)
        rng = np.random if init_seed is None else np.random.RandomState(init_seed)
        self.param = torch.nn.Parameter(torch.zeros(batch, n, rank))
        self.register_buffer('eye', torch.eye(n, rank).unsqueeze(0).repeat(batch, 1, 1))
        if handle_det_sign_pos and rank == n and n % 2 == 1:
            self.register_buffer('negate_last_col', torch.ones(1, 1, n))
            self.negate_last_col[-1] = -1
        with torch.no_grad():
            if init_mode == 'eye':
                self.param.zero_()
            elif init_mode in ('qr_eye_randn', 'qr_randn'):
                if init_mode == 'qr_eye_randn':
                    tmp = (
                        self.eye +
                        init_std * torch.from_numpy(rng.randn(batch, n, rank)).clamp(min=-1, max=1)
                    )
                else:
                    tmp = torch.from_numpy(rng.randn(batch, n, rank))
                param = []
                for m in tmp:
                    q, r = torch.linalg.qr(m)
                    if init_qr_mul_sgn_diag_r:
                        q *= r.diag().sign().view(1, rank)
                    hh = q.geqrf()[0].unsqueeze(0)
                    param.append(hh)
                param = torch.cat(param, dim=0)
                self.param.copy_(param)
            else:
                raise ValueError(f'Unknown orthogonal initialization {init_mode}')

    def forward(self):
        param = self.param.tril(diagonal=-1) + self.eye
        param_norm = 2 / (param * param).sum(dim=1)
        thin = householder_product(param, param_norm)
        if hasattr(self, 'negate_last_col'):
            thin = thin * self.negate_last_col
        if not self.is_thin:
            return thin.permute(0, 2, 1)
        return thin

    def dof(self):
        return self.num_param
