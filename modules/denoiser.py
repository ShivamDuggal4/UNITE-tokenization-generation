import numpy as np
import torch as th
from scipy.stats import norm

from modules.denoiser_utils import path
from modules.denoiser_utils.integrators import ode, sde


class Transport:
    def __init__(
        self,
        train_eps,
        sample_eps,
        use_cosine_loss=False,
        use_lognorm=False,
        partitial_train=None,
        partial_ratio=1.0,
        shift_lg=False,
        lognorm_mu=0.0,
        lognorm_sigma=1.0,
    ):
        
        self.path_sampler = path.ICPlan()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.use_cosine_loss = use_cosine_loss
        self.use_lognorm = use_lognorm
        self.partitial_train = partitial_train
        self.partial_ratio = partial_ratio
        self.shift_lg = shift_lg
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma

    def prior_logp(self, z):
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(self, train_eps, sample_eps, *, diffusion_form="SBDM", sde=False, reverse=False, eval=False, last_step_size=0.0):
        t0, t1 = 0, 1
        eps = train_eps if not eval else sample_eps
        if sde:
            t0 = eps if (diffusion_form == "SBDM" and sde) else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def sample_logit_normal(self, mu, sigma, size=1):
        samples = norm.rvs(loc=mu, scale=sigma, size=size)
        samples = 1 / (1 + np.exp(-samples))
        return th.tensor(samples, dtype=th.float32)

    def sample(self, x1, sp_timesteps=None, shifted_mu=0, timestep_shift=0.0):
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = self.sample_logit_normal(self.lognorm_mu, self.lognorm_sigma, size=x1.shape[0]) * (t1 - t0) + t0
        
        if sp_timesteps is not None: t = th.rand((x1.shape[0],)) * (sp_timesteps[1] - sp_timesteps[0]) + sp_timesteps[0]

        # Timestep shift: t' = alpha*t / (1 + (alpha-1)*t)
        if timestep_shift > 0:
            t = timestep_shift * t / (1.0 + (timestep_shift - 1.0) * t)
        return t.to(x1), x0, x1

    def training_losses(self, model, x1, t=None, model_kwargs=None, sp_timesteps=None, shifted_mu=0):
        if model_kwargs is None:
            model_kwargs = {}
        if t is None:
            t, x0, x1 = self.sample(x1, sp_timesteps, shifted_mu)
        else:
            x0 = th.randn_like(x1)

        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        return {
            "pred": model_output,
            "model_output": model_output,
            "sampled_t": t,
            "xt": xt,
            "x1": x1,
        }

    def get_drift(self):
        def drift_fn(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape
            return model_output

        return body_fn

    def get_score(self):
        return lambda x, t, model, **kw: self.path_sampler.get_score_from_velocity(model(x, t, **kw), x, t)
        

class Sampler:
    """Sampler class for the transport model."""

    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def _get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)

        sde_drift = lambda x, t, model, **kw: (
            self.drift(x, t, model, **kw) + diffusion_fn(x, t) * self.score(x, t, model, **kw)
        )
        return sde_drift, diffusion_fn

    def _get_last_step(self, sde_drift, *, last_step, last_step_size):
        if last_step is None:
            return lambda x, t, model, **kw: x
        elif last_step == "Mean":
            return lambda x, t, model, **kw: x + sde_drift(x, t, model, **kw) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **kw: (
                x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **kw)
            )
        elif last_step == "Euler":
            return lambda x, t, model, **kw: x + self.drift(x, t, model, **kw) * last_step_size
        raise NotImplementedError()

    def sample_sde(self, *, sampling_method="Euler", diffusion_form="SBDM",
                   diffusion_norm=1.0, last_step="Mean", last_step_size=0.04, num_steps=250):
        if last_step is None:
            last_step_size = 0.0
        sde_drift, sde_diffusion = self._get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            diffusion_form=diffusion_form, sde=True, eval=True, reverse=False, last_step_size=last_step_size,
        )
        _sde = sde(sde_drift, sde_diffusion, t0=t0, t1=t1, num_steps=num_steps, sampler_type=sampling_method)
        last_step_fn = self._get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            return xs

        return _sample

    def sample_ode(self, *, sampling_method="dopri5", num_steps=50, atol=1e-6,
                   rtol=1e-3, reverse=False, timestep_shift=0.0):
        if reverse:
            drift = lambda x, t, model, **kw: self.drift(x, th.ones_like(t) * (1 - t), model, **kw)
        else:
            drift = self.drift
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=reverse, last_step_size=0.0,
        )
        _ode = ode(
            drift=drift, t0=t0, t1=t1, sampler_type=sampling_method,
            num_steps=num_steps, atol=atol, rtol=rtol, timestep_shift=timestep_shift,
        )
        return _ode.sample

    def sample_ode_likelihood(self, *, sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3):
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=False, last_step_size=0.0,
        )
        _ode = ode(
            drift=_likelihood_drift, t0=t0, t1=t1, sampler_type=sampling_method,
            num_steps=num_steps, atol=atol, rtol=rtol, timestep_shift=0.0,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            return prior_logp - delta_logp, drift

        return _sample_fn
