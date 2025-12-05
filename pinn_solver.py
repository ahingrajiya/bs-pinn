import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast

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
        
        with torch.no_grad():
            residuals, vega = adaptive_fn(S_cand, t_cand, sigma_cand)
        
        n_error = int(n_target * 0.30)
        _, idx_error = torch.topk(residuals.flatten().abs(), n_error)
        
        n_vega = int(n_target * 0.10)
        _, idx_vega = torch.topk(vega.flatten().abs(), n_vega)
        
        n_random = n_target - n_error - n_vega
        idx_random = torch.randint(0, n_candidates, (n_random,), device=self.device)
        
        indices = torch.cat([idx_error, idx_vega, idx_random])
        
        return S_cand[indices].detach().clone(), t_cand[indices].detach().clone(), sigma_cand[indices].detach().clone()

    def get_batch(self, n_points, epoch, total_epochs, adaptive_fn=None):
        progress = min(epoch / (total_epochs * 0.8), 1.0)
        min_vol, max_vol = self._get_sigma_range(progress)
        
        if adaptive_fn is None:
            return self._sample_sobol(n_points, min_vol, max_vol)
        else:
            return self._sample_adaptive(n_points, min_vol, max_vol, adaptive_fn)


class PINNSolver:
    def __init__(self, model, K, r, T, s_max_multiplier=4.0):
        self.model = model
        self.device = next(model.parameters()).device
        self.K = K
        self.r = r
        self.T = T
        self.s_max = s_max_multiplier * self.K

        initial_sigma_guess = 0.2
        self.log_sigma = nn.Parameter(torch.log(torch.tensor([initial_sigma_guess], device=self.device)))
        
        self.optimizer_pinn = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_pinn, 'min', factor=0.5, patience=1000)
        self.softplus_beta = 1.0
        
        self.data_gen = OptionDataGenerator(self.device, T, self.s_max)

    def _normalize_inputs(self, S, t, sigma, r):
       
        m = S / self.K
        m_norm = (m - 1.0) / 0.5  
        t_norm = 2 * (t / self.T) - 1
        sigma_norm = 2 * ((sigma - 0.05) / (1.0 - 0.05)) - 1
        r_norm = (r / 0.1) * 2 - 1
        
        m_norm.requires_grad_(S.requires_grad)
        t_norm.requires_grad_(t.requires_grad)
        sigma_norm.requires_grad_(sigma.requires_grad)
        r_norm.requires_grad_(r.requires_grad)
        
        return m_norm, t_norm, sigma_norm, r_norm

    def compute_pde_loss_and_vega(self, S, t, sigma):
        S.requires_grad_(True)
        t.requires_grad_(True)
        sigma.requires_grad_(True)
        r_tensor = torch.full_like(S, self.r)

        m_norm, t_norm, sigma_norm, r_norm = self._normalize_inputs(S, t, sigma, r_tensor)
        
        v_pred = F.softplus(self.model(torch.cat([m_norm, t_norm, sigma_norm, r_norm], dim=1)), beta=self.softplus_beta)

        grads = torch.autograd.grad(v_pred.sum(), [m_norm, t_norm], create_graph=True)
        v_m_norm, v_t_norm = grads[0], grads[1]
        
        v_m = v_m_norm * 2.0 
        v_t = v_t_norm * (2.0 / self.T)
        
        v_m_norm_2 = torch.autograd.grad(v_m.sum(), m_norm, create_graph=True)[0]
        v_mm = v_m_norm_2 * 2.0
        
        m = S / self.K
        pde_residual = v_t + 0.5 * (sigma**2) * (m**2) * v_mm + self.r * m * v_m - self.r * v_pred

        v_sigma = torch.autograd.grad(v_pred.sum(), sigma, create_graph=True)[0]

        return pde_residual, v_sigma

    def compute_boundary_terminal_loss(self, n_boundary, n_terminal, option_type, sigma_val):

        m_terminal = torch.rand(n_terminal, 1, device=self.device) * 2.0
        S_terminal = m_terminal * self.K
        t_terminal = torch.full_like(S_terminal, self.T)
        sigma_terminal = torch.full_like(S_terminal, sigma_val)
        r_terminal = torch.full_like(S_terminal, self.r)

        if option_type == 'call':
            true_payoff = torch.relu(m_terminal - 1.0)
        else:
            true_payoff = torch.relu(1.0 - m_terminal)
            
        m_n, t_n, s_n, r_n = self._normalize_inputs(S_terminal, t_terminal, sigma_terminal, r_terminal)
        pred_payoff = F.softplus(self.model(torch.cat([m_n, t_n, s_n, r_n], dim=1)), beta=self.softplus_beta)
        
        loss_terminal = torch.mean((pred_payoff - true_payoff)**2)

        t_boundary = torch.rand(n_boundary, 1, device=self.device) * self.T
        S_zero = torch.zeros_like(t_boundary)
        sigma_b = torch.full_like(t_boundary, sigma_val)
        r_b = torch.full_like(t_boundary, self.r)
        
        m_n0, t_n0, s_n0, r_n0 = self._normalize_inputs(S_zero, t_boundary, sigma_b, r_b)
        pred_zero = F.softplus(self.model(torch.cat([m_n0, t_n0, s_n0, r_n0], dim=1)), beta=self.softplus_beta)
        
        if option_type == 'call':
             loss_boundary = torch.mean(pred_zero**2)
        else:
             true_val = torch.exp(-self.r * (self.T - t_boundary))
             loss_boundary = torch.mean((pred_zero - true_val)**2)
        
        return loss_terminal + loss_boundary

    def train(self, epochs, n_pde, n_boundary, n_terminal, option_type='call', is_american=False, sobol_phase_epochs=8000):
        loss_history = []
        self.model.train()
        scaler = torch.amp.GradScaler()

        S_batch, t_batch, sigma_batch = self.data_gen.get_batch(n_pde, 0, epochs)

        for epoch in range(epochs):
            self.optimizer_pinn.zero_grad()

            if epoch % 200 == 0 and epoch > 0:
                if epoch < sobol_phase_epochs:
                    S_batch, t_batch, sigma_batch = self.data_gen.get_batch(n_pde, epoch, epochs)
                else:
                    def evaluator_fn(s, t, sig):
                        s_eval = s.detach().clone().requires_grad_(True)
                        t_eval = t.detach().clone().requires_grad_(True)
                        sig_eval = sig.detach().clone().requires_grad_(True)
                        with torch.enable_grad():
                            res, v = self.compute_pde_loss_and_vega(s_eval, t_eval, sig_eval)
                        return res.detach(), v.detach()

                    print(f"--- Epoch {epoch}: Adaptive Resampling ---")
                    S_batch, t_batch, sigma_batch = self.data_gen.get_batch(n_pde, epoch, epochs, adaptive_fn=evaluator_fn)

            S_train = S_batch.detach()
            t_train = t_batch.detach()
            sigma_train = sigma_batch.detach()

            with torch.amp.autocast(device_type=self.device.type):
                pde_residual, V_sigma = self.compute_pde_loss_and_vega(S_train, t_train, sigma_train)
                loss_pde = torch.mean(pde_residual**2)
                
                loss_monotonicity = torch.mean(torch.relu(-V_sigma)**2)

                current_sigma_mean = sigma_train.mean().item() 
                loss_bc = self.compute_boundary_terminal_loss(n_boundary, n_terminal, option_type, current_sigma_mean)

                total_loss = loss_pde + 10.0 * loss_bc + 0.1 * loss_monotonicity

            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer_pinn)
            scaler.update()
            self.scheduler.step(total_loss)
            
            loss_history.append(total_loss.item())

            if epoch % 1000 == 0:
                phase = "Sobol" if epoch < sobol_phase_epochs else "Adaptive"
                print(f"Epoch {epoch}/{epochs} [{phase}] Loss: {total_loss.item():.6f} (PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f})")

        return loss_history

    def predict(self, S, t, sigma):
        self.model.eval()
        S_clone = S.clone().requires_grad_(True)
        t_clone = t.clone().requires_grad_(True)
        sigma_clone = sigma.clone().requires_grad_(True)
        r_tensor = torch.full_like(S_clone, self.r)

        m_norm, t_norm, s_norm, r_norm = self._normalize_inputs(S_clone, t_clone, sigma_clone, r_tensor)
        
        with torch.amp.autocast(device_type="cuda"):
            v_pred = F.softplus(self.model(torch.cat([m_norm, t_norm, s_norm, r_norm], dim=1)), beta=self.softplus_beta)
            V_real = v_pred * self.K

        grads = torch.autograd.grad(V_real.sum(), [S_clone, t_clone, sigma_clone], create_graph=True)
        V_S, V_t, V_sigma = grads
        V_SS = torch.autograd.grad(V_S.sum(), S_clone, create_graph=True)[0]

        greeks = { 'delta': V_S, 'gamma': V_SS, 'theta': -V_t, 'vega': V_sigma }
        return V_real, greeks

    def calibrate(self, market_data, epochs=1000, option_type='call', initial_guess=0.20):
        self.model.eval()
        loss_history, sigma_history = [], []
        S_market = market_data['S'].to(self.device)
        t_market = market_data['t'].to(self.device)
        V_market = market_data['V'].to(self.device)
        
        self.log_sigma = nn.Parameter(torch.log(torch.tensor([initial_guess], device=self.device)))
        self.optimizer_sigma = optim.Adam([self.log_sigma], lr=1e-3) 
        
        for epoch in range(epochs):
            self.optimizer_sigma.zero_grad()
            sigma = torch.exp(self.log_sigma)
            sigma.requires_grad_(True)
            sigma_tiled = sigma.expand_as(S_market)
            
            with torch.amp.autocast(device_type="cuda"):
                V_pred, greeks = self.predict(S_market, t_market, sigma_tiled)

            loss = torch.mean((V_pred - V_market)**2)
            
            loss.backward()
            self.optimizer_sigma.step()
            
            loss_history.append(loss.item())
            sigma_history.append(sigma.item())

            if epoch % 500 == 0:
                print(f"  Epoch {epoch}/{epochs}, Loss: {loss.item():.8f}, Current Sigma: {sigma.item():.4f}")

        final_sigma = torch.exp(self.log_sigma).item()
        return final_sigma, loss_history, sigma_history