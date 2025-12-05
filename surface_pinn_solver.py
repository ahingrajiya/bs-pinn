import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from pinn_solver import PINNSolver 

class SurfacePINNSolver(PINNSolver):
    
    def _normalize_inputs_surface(self, S, t, sigma, r):
        m = S / self.K
        m_norm = (m - 1.0) / 0.5 
        t_norm = 2 * (t / self.T) - 1
        sigma_norm = 2 * ((sigma - 0.05) / (1.0 - 0.05)) - 1
        r_norm = (r / 0.1) * 2 - 1
        return m_norm, t_norm, sigma_norm, r_norm

    def calibrate_surface(self, surface_data, epochs=5000, learning_rate=0.01, reg_weight=0.1):
        self.model.eval()
        
        S_flat = surface_data['S'].to(self.device)
        t_flat = surface_data['t'].to(self.device)
        V_market = surface_data['V'].to(self.device)
        r_flat = torch.full_like(S_flat, self.r)
        
        tau = self.T - t_flat
        intrinsic_val = torch.relu(S_flat - self.K * torch.exp(-self.r * tau))
        V_target_time = V_market - intrinsic_val

        n_points = S_flat.shape[0]
        grid_side = int(np.sqrt(n_points))
        is_2d_grid = (grid_side * grid_side == n_points)

        log_sigma_surface = nn.Parameter(
            torch.full((n_points, 1), np.log(0.30), device=self.device, requires_grad=True)
        )
        
        optimizer = optim.Adam([log_sigma_surface], lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)
        
        print(f"--- Calibrating Surface (Stabilized) | Reg={reg_weight} | LR={learning_rate} ---")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            sigma_batch = torch.exp(log_sigma_surface)
            
            S_grad = S_flat.clone().detach().requires_grad_(True)
            m_n, t_n, s_n, r_n = self._normalize_inputs_surface(S_grad, t_flat, sigma_batch, r_flat)
            
            v_pred = F.softplus(self.model(torch.cat([m_n, t_n, s_n, r_n], dim=1)), beta=self.softplus_beta)
            V_pred_dollar = v_pred * self.K
            V_pred_time = V_pred_dollar - intrinsic_val
            
            V_sum = V_pred_dollar.sum()
            vega = torch.autograd.grad(V_sum, log_sigma_surface, create_graph=True)[0]
            
            moneyness = S_flat / self.K
            mask = (torch.abs(vega.detach()) > 1e-5) & (moneyness < 1.3)
            mask = mask.float()

            fit_loss = torch.mean(mask * (V_pred_time - V_target_time)**2)
            
            if is_2d_grid:
                sig_grid = log_sigma_surface.view(grid_side, grid_side)
                d_x = torch.abs(sig_grid[1:, :] - sig_grid[:-1, :])
                d_y = torch.abs(sig_grid[:, 1:] - sig_grid[:, :-1])
                smooth_loss = torch.mean(d_x) + torch.mean(d_y)
            else:
                smooth_loss = torch.mean(torch.abs(log_sigma_surface[1:] - log_sigma_surface[:-1]))
                
            total_loss = fit_loss + reg_weight * smooth_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([log_sigma_surface], max_norm=1.0)
            
            optimizer.step()
            scheduler.step(fit_loss)
            
            if epoch % 500 == 0:
                active_cnt = mask.sum().item()
                mean_vol = sigma_batch.mean().item()
                print(f"Ep {epoch}: Fit {fit_loss.item():.6f} | Smooth {smooth_loss.item():.6f} | Vol {mean_vol:.2f}")
                
        return torch.exp(log_sigma_surface).detach().cpu().numpy()