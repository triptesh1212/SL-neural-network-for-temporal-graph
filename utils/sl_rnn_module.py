import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import partial

from utils.sl_module import sl_exact_step


def cbrt(x):
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


class SLRNN(torch.nn.Module):
    def __init__(self, args, nin):
        super().__init__()

        self.args = args
        
        self.dt = getattr(args, 'dt')
        self.nlayer = args.nlayer
        self.p = args.p
        self.tol = getattr(args, 'tol')

        # Expect batched input of shape (B, T, F)
        self.enc = nn.Linear(nin, args.nhid, dtype=torch.cfloat)
        self.dec = nn.Linear(args.nhid, 1, dtype=torch.cfloat)

        self.zeta = args.zeta
        self.nu = args.nu
        self.activation = partial(F.leaky_relu, negative_slope=args.leaky_relu_slope)


        self.track = []
        self.implicit_r_tol_break = False

    def solve_implicit_r(self, r_prev, max_iter=20): #|z|**p
        r_next = r_prev.clone()
        
        for _ in range(max_iter):
            residual = r_next - r_prev - self.dt * (self.zeta.real*r_next - self.nu.real * r_next**(self.p+1))
            deriv = 1 -self.dt * (self.zeta.real - (self.p+1)*self.nu.real*r_next**self.p)
            update = residual / deriv
            r_next = r_next - update

            if torch.max(torch.abs(update)) < self.tol:
                # print(_)
                return r_next
        else:
            print('Implicit r early exit: max_iter reached', flush=True)
            self.implicit_r_tol_break = True
            return r_next
        
    def solve_explicit_inverse_r(self, r_prev):
        p = (1 - self.dt * self.zeta.real) / (3 * self.nu.real * self.dt)
        q = -r_prev / (2 * self.nu.real * self.dt)
        
        delta = q**2 + p**3

        if torch.all(delta > 0): # there is only one real root
            delta = delta**0.5
            u1 = cbrt(-q + delta)
            u2 = cbrt(-q - delta)

            root1 = u1 + u2
            return root1
        else:
            raise NotImplementedError('delta < 0 not implemented yet')

    def cartesian_implicit_step(self, z, tol=1e-1, max_iter=20):
        x, y = z.real, z.imag
        
        for _ in range(max_iter):
            x3y = x**2 + 3*y**2
            y3x = 3*x**2 + y**2
            xy2 = 2*x*y

            J11 = 1 - self.dt * (self.zeta.real - self.nu.real * y3x + self.nu.imag * xy2)
            J12 = self.dt * (self.zeta.imag + self.nu.real * xy2 - self.nu.imag * x3y)
            J21 = self.dt * (-self.zeta.imag + self.nu.real * xy2 + self.nu.imag * y3x)
            J22 = 1 - self.dt * (self.zeta.real - self.nu.real * x3y - self.nu.imag * xy2)

            jac = torch.stack([
                torch.stack([J11, J12], axis=-1),
                torch.stack([J21, J22], axis=-1)
            ], axis=-2)

            F = x - z.real - self.dt * (self.zeta.real*x - self.zeta.imag*y - (self.nu.real*x - self.nu.imag*y) * (x**2 + y**2))
            G = y - z.imag - self.dt * (self.zeta.imag*x + self.zeta.real*y - (self.nu.real*y + self.nu.imag*x) * (x**2 + y**2))
            RHS = torch.stack([F, G], axis=-1)[..., None]

            delta = torch.linalg.solve(jac, RHS).squeeze()

            x = x - delta[..., 0]
            y = y - delta[..., 1]

            if torch.max(torch.abs(delta)) < tol:
                z_new = x + 1j*y
                return z_new
        else:
            print('Implicit early exit: max_iter reached')
            z_new = x + 1j*y
            return z_new


    def ode_step(self, x):
        theta = torch.angle(x)
        r = torch.abs(x)
        r = self.solve_implicit_r(r) # Newton
        # r = self.solve_explicit_inverse_r(r)
        theta = theta + self.dt * (self.zeta.imag - self.nu.imag * r**2)
        x_new = torch.polar(r, theta)

        # x_new = self.cartesian_implicit_step(x)

        return x_new

        agg = self.activation(conv(torch.stack([r, theta], dim=-1), edge_index))
        x_new = x_new + self.k*self.dt*agg

        return r_new, theta_new
    
    def _encode_sequence(self, x):
        x = x.to(torch.cfloat)
        x = torch.view_as_complex(self.activation(torch.view_as_real(self.enc(x))))  # (B, T, H)

        h = x[:, 0, :]

        for i in range(1, x.shape[1]):
            h = self.ode_step(h)
            h = h + x[:, i, :]

        return h
    
    def _decode_sequence(self, h, length=10):
        outputs = []

        for _ in range(length):
            h = self.ode_step(h)
            out = self.dec(h)
            outputs.append(out.real)  # (B, 1)

        # (B, length)
        outputs = torch.stack(outputs, dim=1).squeeze(-1)
        return outputs
    
    def forward(self, x, length=10):
        # x: (B, T, F) real float32; We'll cast to complex inside encoder
        h = self._encode_sequence(x)
        x = self._decode_sequence(h, length=length)
        return x


class SLSeq(nn.Module):
    def __init__(
        self,
        d_model: int,
        dt: float = 1.0,
        p: int = 2,
        tol: float = 1e-5,
        zeta_real: float = 0.04,
        zeta_imag: float = 0.5,
        nu_real: float = 1.0,
        nu_imag: float = 0.0,
        leaky_relu_slope: float = 0.1,
        use_hid_enc: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.dt = dt
        self.p = p
        self.tol = tol
        self.use_hid_enc = use_hid_enc
        self.activation = partial(F.leaky_relu, negative_slope=leaky_relu_slope)

        self.enc = nn.Linear(d_model, d_model, dtype=torch.cfloat)
        self.dec = nn.Linear(d_model, d_model, dtype=torch.cfloat)
        self.h0 = nn.Parameter(torch.randn(d_model, dtype=torch.cfloat) * 0.01)

        self.zeta = nn.Parameter(
            torch.full((d_model,), complex(zeta_real, zeta_imag), dtype=torch.cfloat)
        )
        self.nu = nn.Parameter(
            torch.full((d_model,), complex(nu_real, nu_imag), dtype=torch.cfloat)
        )

        if use_hid_enc:
            self.hid_enc = nn.Linear(d_model, d_model, dtype=torch.cfloat)

    def ode_step(self, h):
        # Same closed-form SL flow as SL-TGAT (sl_exact_step); clamp Re(ζ), Re(ν) for stability
        zeta = torch.complex(self.zeta.real.clamp(min=1e-4), self.zeta.imag)
        nu = torch.complex(self.nu.real.clamp(min=1e-4), self.nu.imag)
        h_new = sl_exact_step(h, zeta, nu, self.dt)
        if self.use_hid_enc:
            h_new = h_new + 0.1 * torch.view_as_complex(
                self.activation(torch.view_as_real(self.hid_enc(h_new)))
            )
        return h_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) real
        u = torch.view_as_complex(
            self.activation(torch.view_as_real(self.enc(x.to(torch.cfloat))))
        )
        B, L, _ = u.shape
        h = self.h0.unsqueeze(0).expand(B, -1)
        outputs = []
        for t in range(L):
            h = self.ode_step(h)
            h = h + u[:, t, :]
            outputs.append(self.dec(h).real)
        return torch.stack(outputs, dim=1)