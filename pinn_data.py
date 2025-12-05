import torch
import numpy as np

class OptionDataGenerator:

    def __init__(self, device, T, s_max, seed=42):
        self.device = device
        self.T = T
        self.s_max = s_max
        
        self.sobol_engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
        
        self.initial_vol_range = [0.15, 0.45]
        self.final_vol_range = [0.05, 1.0] 
        
    def _get_sigma_range(self, progress):
        curr_min = self.initial_vol_range[0] * (1 - progress) + self.final_vol_range[0] * progress
        curr_max = self.initial_vol_range[1] * (1 - progress) + self.final_vol_range[1] * progress
        return curr_min, curr_max

    def get_batch(self, n_points, epoch, total_epochs, adaptive_fn=None):
        
        progress = min(epoch / (total_epochs * 0.8), 1.0) 
        min_vol, max_vol = self._get_sigma_range(progress)
        
        if adaptive_fn is None:
            return self._sample_sobol(n_points, min_vol, max_vol)
        else:
            return self._sample_adaptive(n_points, min_vol, max_vol, adaptive_fn)

    def _sample_sobol(self, n, min_vol, max_vol):
        quasi = self.sobol_engine.draw(n).to(self.device)
        
        S = quasi[:, 0:1] * self.s_max
        
        t = quasi[:, 1:2] * self.T
        
        sigma = min_vol + (max_vol - min_vol) * quasi[:, 2:3]
        
        S = torch.clamp(S, 1e-4, self.s_max)
        t = torch.clamp(t, 1e-4, self.T)

        return S.detach(), t.detach(), sigma.detach()
    
    def _sample_adaptive(self, n_target, min_vol, max_vol, adaptive_fn):
        n_candidates = n_target * 5
        
        S_cand, t_cand, sigma_cand = self._sample_sobol(n_candidates, min_vol, max_vol)
        
        residuals, vega = adaptive_fn(S_cand, t_cand, sigma_cand)
        
        n_error = int(n_target * 0.30)
        _, idx_error = torch.topk(residuals.flatten().abs(), n_error)
        
        n_vega = int(n_target * 0.10)
        _, idx_vega = torch.topk(vega.flatten().abs(), n_vega)
        
        n_random = n_target - n_error - n_vega
        idx_random = torch.randint(0, n_candidates, (n_random,), device=self.device)
        
        indices = torch.cat([idx_error, idx_vega, idx_random])
        return S_cand[indices], t_cand[indices], sigma_cand[indices]