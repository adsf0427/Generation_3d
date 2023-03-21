import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import argparse
from torch.distributions import Normal

from utils.file_utils import *
from utils.visualize import *
from model.transformer_generation import PointDiffusionTransformer, CLIPImagePointDiffusionTransformer
import torch.distributed as dist
from datasets.shapenet_data_pc import ShapeNet15kPointClouds
import clip
import torch.nn.functional as F
import yaml
from model.clip_model import CLIPPC

'''
some utils
'''

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate(vertices, faces):
    '''
    vertices: [numpoints, 3]
    '''
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:, [1, 2, 0]].dot(M).dot(N).dot(K), faces[:, [1, 2, 0]]
    return v, f


def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f


def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2)
                          for p in net.parameters() if p.grad!=None))
    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)


'''
models
'''


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2)**2 * torch.exp(-logvar2))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(
        torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(
        torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < 0.001, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type,
                 parameterization="eps", clip_ckpt=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config_dir = '/home/internc/CLIP_PC/models/configs/PC.yaml'
        with open(config_dir) as fin:
            config = yaml.safe_load(fin)["PC-B"]
        if clip_ckpt:
            checkpoint = torch.load(clip_ckpt)
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model'):
                    state_dict[k[6:]] = v
            self.clip_model = CLIPPC(**config)
            self.clip_model.load_state_dict(state_dict)
            self.clip_model.to(device)
            self.clip_model = self.clip_model.eval()
            self.clip_model.train = disabled_train
            for param in self.clip_model.parameters():
                param.requires_grad_(False)

        else :
            self.clip_model = None


        self.parameterization = parameterization
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        # computations here in float64 for accuracy
        self.np_betas = betas = betas.astype(np.float64)
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(
            np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(
            1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(
            1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * \
            torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (
            1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(
            x_start.device), t, x_start.shape) * x_start
        variance = self._extract(
            1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod.to(
                x_start.device), t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2.to(
                x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(
            self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, data, t, texts, clip_denoised: bool, return_pred_xstart: bool):
        # print("debug p_mean_variance", texts)
        model_output = denoise_fn(data, t, texts)

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(
                model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(
                model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            if self.parameterization == "eps":
                x_recon = self._predict_xstart_from_eps(
                    data, t=t, eps=model_output)
            elif self.parameterization == "x0":
                x_recon = model_output

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod.to(
                x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, texts, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, texts=texts, clip_denoised=clip_denoised,
                                                                              return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(
            1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * \
            torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, denoise_fn, shape, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False, texts=None):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        if texts == None:
            texts = ['car' for _ in range(shape[0])]
        # print("p_sample_loop", texts[0])
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64,
                             device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, texts=texts, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq,
                                 noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps = self.num_timesteps if not keep_running else len(
            self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        texts = ['car' for _ in range(shape[0])]
        for t in reversed(range(0, total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64,
                             device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, texts=texts, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=data_start, x_t=data_t, t=t)
        texts = ['car' for _ in range(data_t.shape[0])]
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, texts=texts, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped,
                       model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None, texts=None):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = data_start

        if noise is None:
            noise = torch.randn(
                data_start.shape, dtype=data_start.dtype, device=data_start.device)
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            model_out = denoise_fn(data_t, t,  texts=texts)
            assert data_t.shape == data_start.shape
            assert model_out.shape == torch.Size([B, D, N])
            assert model_out.shape == data_start.shape
            losses = ((target - model_out) **
                      2).mean(dim=list(range(1, len(data_start.shape))))
            if self.clip_model:
                assert self.parameterization == "x0"
                pc_inputs = model_out
                text_inputs = clip.tokenize(texts).to(device=data_start.device)
                
                pc_features = self.clip_model.encode_pc(pc_inputs)
                text_features = self.clip_model.encode_text(text_inputs)

                pc_features = F.normalize(pc_features, dim=1)           
                text_features = F.normalize(text_features, dim=1)   
                image_logits = pc_features @ text_features.t() * self.clip_model.logit_scale.exp()
                ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
                clip_loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
                return losses, clip_loss
                


        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses, None

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64,
                             device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_ = torch.zeros([B, T], device=x_start.device), torch.zeros([
                B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64,
                                  device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start) **
                             2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape == torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None] == torch.arange(
                    T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + \
                    new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([
                                                                                       B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                total_bpd_b.shape == prior_bpd_b.shape == torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()



class ModelTransformer(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str, parameterization: str, uncondition: bool = True, clip_ckpt: str = None
                 ):
        super(ModelTransformer, self).__init__()
        self.diffusion = GaussianDiffusion(
            betas, loss_type, model_mean_type, model_var_type, parameterization=parameterization, clip_ckpt=clip_ckpt)
        self.uncondition = uncondition

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if uncondition:
            print("uncondition")
            self.model = PointDiffusionTransformer(
                device=device, dtype=torch.float32, time_token_cond=True
            )
        else :
            print("condition")
            self.model = CLIPImagePointDiffusionTransformer(
                device=device, dtype=torch.float32, token_cond=True, time_token_cond=True
            )


    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(
            self._denoise, x0, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt': mse_bt
        }

    def _denoise(self, data, t, texts):
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, texts)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None, texts = None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps,
                          size=(B,), device=data.device)

        if noises is not None:
            noises[t != 0] = torch.randn(
                (t != 0).sum(), *noises.shape[1:]).to(noises)

        losses, clip_loss = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, texts = texts)
        assert losses.shape == t.shape == torch.Size([B])
        return losses, clip_loss

    def gen_samples(self, shape, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False,
                    texts=None):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running, texts=texts)

    def gen_sample_traj(self, shape, device, freq, noise_fn=torch.randn,
                        clip_denoised=True, keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(
            b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(
            b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(
            b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


def train(gpu, opt):

    set_seed(opt)

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank = opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

        opt.saveIter = int(opt.saveIter / opt.ngpus_per_node)
        opt.diagIter = int(opt.diagIter / opt.ngpus_per_node)
        opt.vizIter = int(opt.vizIter / opt.ngpus_per_node)



    betas = get_betas(opt.schedule_type, opt.beta_start,
                      opt.beta_end, opt.time_num)

    if opt.use_transformer:
        print(opt.uncondition)
        model = ModelTransformer(opt, betas, opt.loss_type,
                             opt.model_mean_type, opt.model_var_type, parameterization=opt.parameterization ,uncondition=opt.uncondition, clip_ckpt=opt.clip_ckpt)
    else:
        model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)

    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')



    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])


    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)


    model.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.rand(25, 3, 2048, device=device)
        cat = 'a model of airplane'
        texts = [cat for _ in range(x.shape[0])]
        print(texts)

        x_gen_eval = model.gen_samples(new_x_chain(
            x, x.shape[0]).shape, x.device, clip_denoised=False, texts=texts)

    # x_gen_eval = x_gen_eval.repeat(5,1,1)
    visualize_pointcloud_batch('./output/test/%s.png' % cat,
                                x_gen_eval.transpose(1, 2), None, None,
                                None)

    # dist.destroy_process_group()


def main():
    opt = parse_args()

    ''' workaround '''



    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node,
                 args=(opt))
    else:
        train(opt.gpu, opt)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000,
                        help='number of epochs to train for')

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    '''model'''
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    # params
    parser.add_argument('--uncondition', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_transformer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--clip_ckpt', default=None)
    parser.add_argument('--parameterization', default='eps')
    

    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float,
                        default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.998, help='lr decay for EBM')

    parser.add_argument('--model', default='',
                        help="path to model (to continue training)")

    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:19991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single' , choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    '''eval'''
    parser.add_argument('--saveIter', default=20, help='unit: epoch')
    parser.add_argument('--diagIter', default=20, help='unit: epoch')
    parser.add_argument('--vizIter', default=20, help='unit: epoch')
    parser.add_argument('--print_freq', default=20, help='unit: iter')

    parser.add_argument('--manualSeed', default=42,
                        type=int, help='random seed')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()
