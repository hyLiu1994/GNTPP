from .model import *
from ..base_prob_dec import BaseProbDecoder
from . import networks
from . import probability
from .networks import adapters
from omegaconf import OmegaConf, DictConfig
from .utils_train import (
    make_config, make_dataloaders, make_from_cfg
)
from . import BFN_Decoder

class TimeIntervalBayesianFlowLoss(Loss):
    def __init__(
        self,
        bayesian_flow: CtsBayesianFlow,
        distribution_factory: Union[CtsDistributionFactory, DiscreteDistributionFactory],
        min_loss_variance: float = -1,
        noise_pred: bool = True,
    ):
        super().__init__()
        self.bayesian_flow = bayesian_flow
        self.distribution_factory = distribution_factory
        self.min_loss_variance = min_loss_variance
        self.C = -0.5 * math.log(bayesian_flow.min_variance)
        self.noise_pred = noise_pred
        if self.noise_pred:
            self.distribution_factory.log_dev = False
            self.distribution_factory = PredDistToDataDistFactory(
                self.distribution_factory, self.bayesian_flow.min_variance
            )

    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t) -> Tensor:
        output_params = sandwich(output_params)
        t = t.flatten(start_dim=1).float()
        posterior_var = torch.pow(self.bayesian_flow.min_variance, t)
        flat_target = data.flatten(start_dim=1)
        pred_dist = self.distribution_factory.get_dist(output_params, input_params, t)
        pred_mean = pred_dist.mean
        mse_loss = (pred_mean - flat_target).square()
        if self.min_loss_variance > 0:
            posterior_var = posterior_var.clamp(min=self.min_loss_variance)
        loss = self.C * mse_loss / posterior_var
        return loss

    def discrete_time_loss(
        self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor, n_steps: int, n_samples=10
    ) -> Tensor:
        output_params = sandwich(output_params)
        t = t.flatten(start_dim=1).float()
        output_dist = self.distribution_factory.get_dist(output_params, input_params, t)
        if hasattr(output_dist, "probs"):  # output distribution is discretized normal
            flat_target = data.flatten(start_dim=1)
            t = t.flatten(start_dim=1)
            i = t * n_steps + 1  # since t = (i - 1) / n
            _, alpha = self.bayesian_flow.get_alpha(i, n_steps)
            _, sender_dist = self.bayesian_flow.get_sender_dist(flat_target, alpha)
            receiver_mix_wts = sandwich(output_dist.probs())
            receiver_mix_dist = D.Categorical(probs=receiver_mix_wts, validate_args=False)
            receiver_components = D.Normal(
                output_dist.class_centres, (1.0 / alpha.sqrt()).unsqueeze(-1), validate_args=False
            )
            receiver_dist = D.MixtureSameFamily(receiver_mix_dist, receiver_components, validate_args=False)
            y = sender_dist.sample(torch.Size([n_samples]))
            loss = (
                (sender_dist.log_prob(y) - receiver_dist.log_prob(y))
                .mean(0)
                .flatten(start_dim=1)
                .mean(1, keepdims=True)
            )
        else:  # output distribution is normal
            pred_mean = output_dist.mean
            flat_target = data.flatten(start_dim=1)
            mse_loss = (pred_mean - flat_target).square()
            i = t * n_steps + 1
            _, alpha = self.bayesian_flow.get_alpha(i, n_steps)
            loss = alpha * mse_loss / 2
        return n_steps * loss

    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        output_params = sandwich(output_params)
        flat_data = data.flatten(start_dim=1)
        t = torch.ones_like(data).flatten(start_dim=1).float()
        output_dist = self.distribution_factory.get_dist(output_params, input_params, t)

        if hasattr(output_dist, "probs"):  # output distribution is discretized normal
            reconstruction_loss = -output_dist.log_prob(flat_data)
        else:  # output distribution is normal, but we use discretized normal to make results comparable (see Sec. 7.2)
            if self.bayesian_flow.min_variance == 1e-3:  # used for 16 bin CIFAR10
                noise_dev = 0.7 * math.sqrt(self.bayesian_flow.min_variance)
                num_bins = 16
            else:
                noise_dev = math.sqrt(self.bayesian_flow.min_variance)
                num_bins = 256
            mean = output_dist.mean.flatten(start_dim=1)
            final_dist = D.Normal(mean, noise_dev)
            final_dist = DiscretizedCtsDistribution(final_dist, num_bins, device=t.device, batch_dims=mean.ndim - 1)
            reconstruction_loss = -final_dist.log_prob(flat_data)
        return reconstruction_loss

class TypeBayesianFlowLoss(Loss):
    def __init__(
        self,
        bayesian_flow: DiscreteBayesianFlow,
        distribution_factory: DiscreteDistributionFactory,
    ):
        super().__init__()
        self.bayesian_flow = bayesian_flow
        self.distribution_factory = distribution_factory
        self.K = self.bayesian_flow.n_classes

    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t) -> Tensor:
        flat_output = sandwich(output_params)
        pred_probs = self.distribution_factory.get_dist(flat_output).probs()
        flat_target = data.flatten(start_dim=1)
        if self.bayesian_flow.discretize:
            flat_target = float_to_idx(flat_target, self.K)
        tgt_mean = torch.nn.functional.one_hot(flat_target.long(), self.K)
        kl = self.K * ((tgt_mean - pred_probs).square()).sum(-1)
        t = t.flatten(start_dim=1).float()
        loss = t * (self.bayesian_flow.max_sqrt_beta**2) * kl
        return loss

    def discrete_time_loss(
        self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor, n_steps: int, n_samples=10
    ) -> Tensor:
        flat_target = data.flatten(start_dim=1)
        if self.bayesian_flow.discretize:
            flat_target = float_to_idx(flat_target, self.K)
        i = t * n_steps + 1
        alpha, _ = self.bayesian_flow.get_alpha(i, n_steps).flatten(start_dim=1)
        sender_dist, _ = self.bayesian_flow.get_sender_dist(flat_target, alpha)

        flat_output = sandwich(output_params)
        receiver_mix_wts = self.distribution_factory.get_dist(flat_output).probs()
        receiver_mix_dist = D.Categorical(probs=receiver_mix_wts.unsqueeze(-2))
        classes = torch.arange(self.K, device=flat_target.device).long().unsqueeze(0).unsqueeze(0)
        receiver_components, _ = self.bayesian_flow.get_sender_dist(classes, alpha.unsqueeze(-1))
        receiver_dist = D.MixtureSameFamily(receiver_mix_dist, receiver_components)

        y = sender_dist.sample(torch.Size([n_samples]))
        loss = n_steps * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).sum(-1).mean(1, keepdims=True)
        return loss

    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        flat_outputs = sandwich(output_params)
        flat_data = data.flatten(start_dim=1)
        output_dist = self.distribution_factory.get_dist(flat_outputs)
        return -output_dist.log_prob(flat_data)

class TPPBayesianFlow(BayesianFlow):
    def __init__(
        self,
        n_classes: int,
        min_sqrt_beta: float = 1e-10,
        discretize: bool = False,
        epsilon: float = 1e-6,
        max_sqrt_beta: float = 1,
        min_variance: float = 1e-6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.min_sqrt_beta = min_sqrt_beta
        self.discretize = discretize
        self.epsilon = epsilon
        self.max_sqrt_beta = max_sqrt_beta
        self.uniform_entropy = math.log(self.n_classes)
        self.min_variance = min_variance
    
    @torch.no_grad()
    def forward(self, data_event_type: Tensor, data_timeinterval: Tensor, t: Tensor):
        if self.discretize:
            data_event_type = float_to_idx(data_event_type, self.n_classes)
        sqrt_beta = self.t_to_sqrt_beta(t.clamp(max=1 - self.epsilon))
        lo_beta = sqrt_beta < self.min_sqrt_beta
        sqrt_beta = sqrt_beta.clamp(min=self.min_sqrt_beta)
        beta = sqrt_beta.square().unsqueeze(-1)
        logits = self.count_sample(data_event_type, beta)
        probs = F.softmax(logits, -1)
        probs = torch.where(lo_beta.unsqueeze(-1), torch.ones_like(probs) / self.n_classes, probs)
        if self.n_classes == 2:
            probs = probs[..., :1]
            probs = probs.reshape_as(data_event_type)
        input_params_type = (probs,)

        post_var = torch.pow(self.min_variance, t)
        alpha_t = 1 - post_var
        mean_mean = alpha_t * data_timeinterval
        mean_var = alpha_t * post_var
        mean_std_dev = mean_var.sqrt()
        noise = torch.randn(mean_mean.shape, device=mean_mean.device)
        mean = mean_mean + (mean_std_dev * noise)
        # We don't need to compute the variance because it is not needed by the network, so set it to None
        input_params_timeinterval = (mean, None)
        return input_params_type, input_params_timeinterval

    def t_to_sqrt_beta(self, t):
        return t * self.max_sqrt_beta

    def count_dist(self, x, beta=None):
        mean = (self.n_classes * F.one_hot(x.long(), self.n_classes)) - 1
        std_dev = math.sqrt(self.n_classes)
        if beta is not None:
            mean = mean * beta
            std_dev = std_dev * beta.sqrt()
        return D.Normal(mean, std_dev, validate_args=False)

    def count_sample(self, x, beta):
        return self.count_dist(x, beta).rsample()

    @torch.no_grad()
    def get_prior_input_params(self, data_shape_type: tuple, data_shape_timeinterval: tuple, device: torch.device):
        return (torch.ones(*data_shape_type, self.n_classes, device=device) / self.n_classes,), (torch.zeros(*data_shape_timeinterval, device=device), 1.0)

    @torch.no_grad()
    def params_to_net_inputs(self, params_type, params_timeinterval) -> Tensor:
        params_type = params_type[0]
        if self.n_classes == 2:
            params_type = params_type * 2 - 1  # We scale-shift here for MNIST instead of in the network like for text
            params_type = params_type[..., :1]

        return params_type, params_timeinterval[0]

    def get_alpha(self, i: Union[int, Tensor], n_steps: int) -> Union[float, Tensor]:
        sigma_1 = math.sqrt(self.min_variance)
        return ((self.max_sqrt_beta / n_steps) ** 2) * (2 * i - 1), (sigma_1 ** (-2 * i / n_steps)) * (1 - sigma_1 ** (2 / n_steps)) 

    def get_sender_dist(self, x_type: Tensor, alpha_type: Union[float, Tensor], x_timeinterval: Tensor, alpha_timeinterval: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        e_x = F.one_hot(x_type.long(), self.n_classes)
        alpha_type = alpha_type.unsqueeze(-1) if isinstance(alpha_type, Tensor) else alpha_type
        dist_type = D.Normal(alpha_type * ((self.n_classes * e_x) - 1), (self.n_classes * alpha_type) ** 0.5)

        dist_timeinterval = D.Normal(x_timeinterval, 1.0 / alpha_timeinterval**0.5)
        return dist_type, dist_timeinterval
    
    def update_input_params(self, input_params_type, y_type: Tensor, alpha_type: float,
                            input_params_timeinterval, y_timeinterval: Tensor, alpha_timeinterval: float):
        new_input_params_type = input_params_type[0] * y_type.exp()
        new_input_params_type /= new_input_params_type.sum(-1, keepdims=True)

        input_mean, input_precision = input_params_timeinterval
        new_precision = input_precision + alpha_timeinterval
        new_mean = ((input_precision * input_mean) + (alpha_timeinterval * y_timeinterval)) / new_precision
        return (new_input_params_type,), (new_mean, new_precision)

class BFNDecoder(BaseProbDecoder):
    def __init__(self, 
                 embed_size,
                 layer_num,
                 event_type_num,
                 mean_log_inter_time: float=0.0,
                 std_log_inter_time: float=1.0, 
                 *args, **kwargs):
        super().__init__(embed_size=embed_size, layer_num=layer_num, event_type_num=event_type_num,
                         mean_log_inter_time=mean_log_inter_time, std_log_inter_time=std_log_inter_time,
                         *args, **kwargs)


        cfg = make_config('models/prob_decoders/bayesian_flow_network/config/BFN_stackoverflow.yaml')
        data_adapters = {
            "input_adapter": make_from_cfg(adapters, cfg.model.input_adapter),
            "output_adapter": make_from_cfg(adapters, cfg.model.output_adapter),
        }
        net = make_from_cfg(networks, cfg.model.net, data_adapters=data_adapters)
        bayesian_flow = make_from_cfg(BFN_Decoder, cfg.model.bayesian_flow)
        distribution_factory_type = make_from_cfg(probability, cfg.model.distribution_factory_type)
        loss_type = make_from_cfg(BFN_Decoder, cfg.model.loss_type, bayesian_flow=bayesian_flow, distribution_factory=distribution_factory_type)
        distribution_factory_timeinterval = make_from_cfg(probability, cfg.model.distribution_factory_timeinterval)
        loss_timeinterval = make_from_cfg(BFN_Decoder, cfg.model.loss_timeinterval, bayesian_flow=bayesian_flow, distribution_factory=distribution_factory_timeinterval)

        self.net = net
        self.bayesian_flow = bayesian_flow
        self.loss_type = loss_type
        self.loss_timeinterval = loss_timeinterval

    @staticmethod
    @torch.no_grad()
    def sample_t(data: Tensor, n_steps: Optional[int]) -> Tensor:
        if n_steps == 0 or n_steps is None:
            t = torch.rand(data.size(0), device=data.device).unsqueeze(-1)
        else:
            t = torch.randint(0, n_steps, (data.size(0),), device=data.device).unsqueeze(-1) / n_steps
        t = (torch.ones_like(data).flatten(start_dim=1) * t).reshape_as(data)
        return t

    def forward(
        self, data_timeinterval: Tensor, data_type: Tensor, seq_onehots: Tensor, history_embedding: Tensor, t: Optional[Tensor] = None, n_steps: Optional[int] = None
    ):
        """
        Compute an MC estimate of the continuous (when n_steps=None or 0) or discrete time KL loss.
        t is sampled randomly if None. If t is not None, expect t.shape == data.shape.
        """
        # batch_size, seq_len = data_timeinterval.shape

        data_timeinterval = data_timeinterval.reshape(-1).unsqueeze(-1)
        data_type = data_type.reshape(-1)
        seq_onehots = seq_onehots.reshape(-1, seq_onehots.shape[-1])
        history_embedding = history_embedding.squeeze(-2)
        history_embedding = history_embedding.reshape(-1, history_embedding.shape[-1])

        data_timeinterval = data_timeinterval[data_type != 0]
        seq_onehots = seq_onehots[data_type != 0]
        history_embedding = history_embedding[data_type != 0]
        data_type = data_type[data_type != 0].unsqueeze(-1)

        t = self.sample_t(data_type, n_steps) if t is None else t
        # sample input parameter flow
        input_params_type, input_params_timeinterval = self.bayesian_flow(data_type, data_timeinterval, t)
        net_inputs_type, net_inputs_timeinterval = self.bayesian_flow.params_to_net_inputs(input_params_type, input_params_timeinterval)

        # compute output distribution parameters
        output_params_type, output_params_timeinterval = self.net(net_inputs_type, net_inputs_timeinterval, history_embedding, t)

        # compute KL loss in float32
        if n_steps == 0 or n_steps is None:
            loss_type = self.loss_type.cts_time_loss(data_type, output_params_type.float(), input_params_type, t)
            loss_timeinterval = self.loss_timeinterval.cts_time_loss(data_timeinterval, output_params_timeinterval.float(), input_params_timeinterval, t)
        else:
            loss_type = self.loss_type.discrete_time_loss(data_type, output_params_type.float(), input_params_type, t, n_steps)
            loss_timeinterval = self.loss_timeinterval.discrete_time_loss(data_timeinterval, output_params_timeinterval.float(), input_params_timeinterval, t, n_steps)

        # loss shape is (batch_size, 1)
        return loss_type.mean(), loss_timeinterval.mean()

    # @torch.inference_mode()
    def compute_reconstruction_loss(self, data_type: Tensor, data_timeinterval: Tensor, history_embedding: Tensor) -> Tensor:
        t = torch.ones_like(data_type).float()
        input_params_type, input_params_timeinterval = self.bayesian_flow(data_type, data_timeinterval, t)
        net_inputs_type, net_inputs_timeinterval = self.bayesian_flow.params_to_net_inputs(input_params_type, input_params_timeinterval)
        output_params_type, output_params_timeinterval = self.net(net_inputs_type, net_inputs_timeinterval, history_embedding, t)
        return self.loss_type.reconstruction_loss(data_type, output_params_type, input_params_type).flatten(start_dim=1).mean() + \
            self.loss_timeinterval.reconstruction_loss(data_timeinterval, output_params_timeinterval, input_params_timeinterval).flatten(start_dim=1).mean()

    # @torch.inference_mode()
    def sample(self, data_shape_type: tuple, data_shape_timeinterval: tuple, history_embedding: Tensor, n_steps: int) -> Tensor:
        device = next(self.parameters()).device
        input_params_type, input_params_timeinterval = self.bayesian_flow.get_prior_input_params(data_shape_type, data_shape_timeinterval, device)
        distribution_factory_type = self.loss_type.distribution_factory
        distribution_factory_timeinterval = self.loss_timeinterval.distribution_factory

        for i in range(1, n_steps):
            t = torch.ones(*data_shape_type, device=device) * (i - 1) / n_steps
            net_input_type, net_input_timeinterval = self.bayesian_flow.params_to_net_inputs(input_params_type, input_params_timeinterval)
            output_params_type, output_params_timeinterval = self.net(net_input_type, net_input_timeinterval, history_embedding, t)
            output_sample_type = distribution_factory_type.get_dist(output_params_type, input_params_type, t).sample()
            output_sample_timeinterval = distribution_factory_timeinterval.get_dist(output_params_timeinterval, input_params_timeinterval, t).sample()
            output_sample_type = output_sample_type.reshape(*data_shape_type)
            output_sample_timeinterval = output_sample_timeinterval.reshape(*data_shape_timeinterval)
            alpha_type, alpha_timeinterval = self.bayesian_flow.get_alpha(i, n_steps)
            y_type, y_timeinterval = self.bayesian_flow.get_sender_dist(output_sample_type, alpha_type, output_sample_timeinterval, alpha_timeinterval).sample()
            input_params_type, input_params_timeinterval = self.bayesian_flow.update_input_params(input_params_type, y_type, alpha_type,
                                                                  input_params_timeinterval, y_timeinterval, alpha_timeinterval)

        t = torch.ones(*data_shape_type, device=device)
        net_input_type, net_input_timeinterval = self.bayesian_flow.params_to_net_inputs(input_params_type, input_params_timeinterval)
        output_params_type, output_params_timeinterval = self.net(net_input_type, net_input_timeinterval, history_embedding, t)
        output_sample_type = distribution_factory_type.get_dist(output_params_type, input_params_type, t).sample()
        output_sample_timeinterval = distribution_factory_timeinterval.get_dist(output_params_timeinterval, input_params_timeinterval, t).sample()
        output_sample_type = output_sample_type.reshape(*data_shape_type)
        output_sample_timeinterval = output_sample_timeinterval.reshape(*data_shape_timeinterval)
        return output_sample_type, output_sample_timeinterval