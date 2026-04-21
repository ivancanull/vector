import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Parameter

class BayesianTransposedConv1D(nn.Module):
    """
    Bayesian Transposed Convolutional Layer for 1D inputs.
    """
    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super(BayesianTransposedConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        self.weight_mu = Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        self.weight_log_sigma = Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        self.bias_mu = Parameter(torch.Tensor(out_channels))
        self.bias_log_sigma = Parameter(torch.Tensor(out_channels))

        self.register_buffer('weight_eps', None)

        if bias:
            self.bias = True
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_log_sigma = Parameter(torch.Tensor(out_channels))
            self.register_buffer('bias_eps', None)
        else:
            self.bias = False
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # initialize the weights and biases
        stdv = 1. / math.sqrt(self.in_channels * self.kernel_size ** 2)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self):
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias:
            self.bias_eps = torch.randn_like(self.bias_log_sigma)
    
    def unfreeze(self):
        self.weight_eps = None
        if self.bias:
            self.bias_eps = None

    def forward(self, input):

        if self.weight_eps is None:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        
        if self.bias:
            if self.bias_eps is None:
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else:
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else:
            bias = None
        return F.conv_transpose1d(input, weight, bias=bias, stride=self.stride, padding=self.padding, output_padding=self.output_padding)


def test_BayesianTransposedConv1D():
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = 32
    input_length = 10

    # Create a random input tensor
    input_tensor = torch.randn(batch_size, in_channels, input_length)

    # Initialize the Bayesian Transposed Conv1D layer
    layer = BayesianTransposedConv1D(0.0, 1.0, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True)

    # Forward pass
    output_tensor = layer(input_tensor)
    
    print("Output shape:", output_tensor.shape)