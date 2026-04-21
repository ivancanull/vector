import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear import BayesianLinear

def calculate_Lout(
    Lin,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    output_padding=0
):
    Lout = (Lin - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return Lout

class MultiOutputBNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, prior_mus, prior_sigmas, output_channels):
        """
        MLP with multiple output channels
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dims (list): List of hidden layer dimensions
            prior_mus (list): List of prior means for each hidden layer
            prior_sigmas (list): List of prior standard deviations for each hidden layer
            output_channels (dict): Dictionary of output specifications
                Format: {channel_name: (output_dim, activation)}
                Example: {'regression': (1, None), 'classification': (10, 'softmax')}
        """
        super(MultiOutputBNN, self).__init__()
        
        # Shared hidden layers
        self.shared_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.shared_layers.append(BayesianLinear(prior_mu=prior_mus[i], prior_sigma=prior_sigmas[i], in_features=dims[i], out_features=dims[i+1]))
            # self.shared_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(0.3))
        
        # Output heads
        self.names = []
        self.output_heads = nn.ModuleDict()
        for name, (out_dim, _) in output_channels.items():
            head = nn.Sequential(
                BayesianLinear(prior_mu=prior_mus[-1], prior_sigma=prior_sigmas[-1], in_features=dims[-1], out_features=dims[-1]),
                nn.Sigmoid(),
                nn.Dropout(0.3),
                BayesianLinear(prior_mu=prior_mus[-1], prior_sigma=prior_sigmas[-1], in_features=dims[-1], out_features=out_dim)
            )
            self.output_heads[name] = head
            self.names.append(name)

        self.activations = {name: activation for name, (_, activation) in output_channels.items()}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for head in self.output_heads.values():
            if isinstance(head[0], nn.Linear):
                nn.init.xavier_normal_(head[0].weight)
                nn.init.constant_(head[0].bias, 0)
    
    def forward(self, x):
        # Process through shared layers
        for layer in self.shared_layers:
            x = layer(x)
        
        # Process through output heads
        outputs = {}
        for name, head in self.output_heads.items():
            out = head(x)
            
            # Apply activation if specified
            if self.activations[name] == 'sigmoid':
                outputs[name] = torch.sigmoid(out)
            elif self.activations[name] == 'softmax':
                outputs[name] = F.softmax(out, dim=1)
            elif self.activations[name] == 'tanh':
                outputs[name] = torch.tanh(out)
            elif self.activations[name] == 'relu':
                outputs[name] = F.relu(out)
            else:
                outputs[name] = out  # No activation (e.g., for regression)
        
        return outputs
    
    def freeze(self):
        """
        Freeze the model parameters.
        """
        for m in self.modules() :
            if isinstance(m, (BayesianLinear)):
                m.freeze()
    
    def unfreeze(self):
        """
        Unfreeze the model parameters.
        """
        for m in self.modules() :
            if isinstance(m, (BayesianLinear)):
                m.unfreeze()

class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_channels):
        """
        MLP with multiple output channels
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dims (list): List of hidden layer dimensions
            prior_mus (list): List of prior means for each hidden layer
            prior_sigmas (list): List of prior standard deviations for each hidden layer
            output_channels (dict): Dictionary of output specifications
                Format: {channel_name: (output_dim, activation)}
                Example: {'regression': (1, None), 'classification': (10, 'softmax')}
        """
        super(MultiOutputMLP, self).__init__()
        
        # Shared hidden layers
        self.shared_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.shared_layers.append(nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            # self.shared_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(0.3))
        
        # Output heads
        self.names = []
        self.output_heads = nn.ModuleDict()
        for name, (out_dim, _) in output_channels.items():
            head = nn.Sequential(
                nn.Linear(dims[-1], dims[-1]),
                nn.Sigmoid(),
                nn.Dropout(0.3),
                nn.Linear(dims[-1], out_dim)
            )
            self.output_heads[name] = head
            self.names.append(name)

        self.activations = {name: activation for name, (_, activation) in output_channels.items()}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for head in self.output_heads.values():
            if isinstance(head[0], nn.Linear):
                nn.init.xavier_normal_(head[0].weight)
                nn.init.constant_(head[0].bias, 0)
    
    def forward(self, x):
        # Process through shared layers
        for layer in self.shared_layers:
            x = layer(x)
        
        # Process through output heads
        outputs = {}
        for name, head in self.output_heads.items():
            out = head(x)
            
            # Apply activation if specified
            if self.activations[name] == 'sigmoid':
                outputs[name] = torch.sigmoid(out)
            elif self.activations[name] == 'softmax':
                outputs[name] = F.softmax(out, dim=1)
            elif self.activations[name] == 'tanh':
                outputs[name] = torch.tanh(out)
            elif self.activations[name] == 'relu':
                outputs[name] = F.relu(out)
            else:
                outputs[name] = out  # No activation (e.g., for regression)
        
        return outputs
    
    def freeze(self):
        """
        Freeze the model parameters.
        """
        return # nothing to freeze in MLP
 
    def unfreeze(self):
        """
        Unfreeze the model parameters.
        """
        return 
    
class MultiOutputTCN(nn.Module):
    """
    A simple neural network with transposed convolution layers.
    This network is designed for tasks upsampling.
    """

    def __init__(self, input_dim, hidden_dims, output_channels, tcns):
        

        super(MultiOutputTCN, self).__init__()
        self.backbone = nn.ModuleList()

        dims = [input_dim] + hidden_dims[0:3]
        for i in range(3):
            self.backbone.append(nn.Linear(in_features=dims[i], out_features=dims[i+1]))
            self.backbone.append(nn.ReLU())
            self.backbone.append(nn.Dropout(0.3))

        dims = hidden_dims[-4:]

        kernel_size, stride, padding, dilation, output_padding = tcns["kernel_size"], tcns["stride"], tcns["padding"], tcns["dilation"], tcns["output_padding"]

        self.upsampling = nn.ModuleList([nn.Unflatten(-1, (-1, 1))])
        for i in range(len(dims) - 1):
            self.upsampling.append(nn.ConvTranspose1d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size[i], stride=stride[i], padding=padding[i], dilation=dilation[i], output_padding=output_padding[i]))
            self.upsampling.append(nn.ReLU())
            self.upsampling.append(nn.Dropout(0.3))

        # L = 1
        # for i in range(len(dims) - 1):
        #     L = calculate_Lout(
        #         Lin=L,
        #         kernel_size=20,
        #         stride=2,
        #         padding=0,
        #         dilation=1,
        #         output_padding=0
        #     )
        self.names = []
        self.output_heads = nn.ModuleDict()
        for name, (out_channel, out_dim) in output_channels.items():
            head = nn.Sequential(
                nn.Conv1d(dims[-1], out_channel, kernel_size=1),
            )
            self.output_heads[name] = head
            self.names.append(name)

        self.activations = {name: activation for name, (_, activation) in output_channels.items()}

    def _init_weights(self):
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        for layer in self.upsampling:
            if isinstance(layer, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        for head in self.output_heads.values():
            if isinstance(head[0], nn.Conv1d):
                nn.init.kaiming_normal_(head[0].weight)
                nn.init.constant_(head[0].bias, 0)
    
    def forward(self, x):
        # Process through shared layers
        for layer in self.backbone:
            x = layer(x)

        for layer in self.upsampling:
            x = layer(x)
        
        # Process through output heads
        outputs = {}
        for name, head in self.output_heads.items():
            out = head(x)
            
            # Apply activation if specified
            if self.activations[name] == 'sigmoid':
                outputs[name] = torch.sigmoid(out)
            elif self.activations[name] == 'softmax':
                outputs[name] = F.softmax(out, dim=1)
            elif self.activations[name] == 'tanh':
                outputs[name] = torch.tanh(out)
            elif self.activations[name] == 'relu':
                outputs[name] = F.relu(out)
            else:
                outputs[name] = out  # No activation (e.g., for regression)
        return outputs

    def freeze(self):
        """
        Freeze the model parameters.
        """
        return # nothing to freeze in MLP
 
    def unfreeze(self):
        """
        Unfreeze the model parameters.
        """
        return   

# Example usage
def test_multi_output_bnn():
    
    # Configuration
    input_dim = 4
    hidden_dims = [256, 128, 64]
    prior_mus = [0.0] * len(hidden_dims)
    prior_sigmas = [0.1] * len(hidden_dims)
    output_channels = {
        'r': (2, None),        # Single value output (no activation)
        'i': (2, None),  # 10-class classification
    }
    
    # Create model
    model = MultiOutputBNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        prior_mus=prior_mus,
        prior_sigmas=prior_sigmas,
        output_channels=output_channels,

    )
    
    # Sample input
    batch_size = 16
    input_data = torch.randn(batch_size, input_dim)
    
    # Forward pass
    outputs = model(input_data)
    for key in outputs.keys():
        key = key + "a"
    for key in outputs.keys():
        print(key)
    for m in model.modules() :
        if isinstance(m, (BayesianLinear)):    # Print outputs
            print(m)
    for channel, output in outputs.items():
        print(f"{channel}: {output.shape}")
    
    # Sample outputs
    print("\nSample regression output:", outputs['r'][0])
    print("Sample classification output:", outputs['i'][0])

def test_transposed_conv_net():

    # Configuration
    input_dim = 3
    hidden_dims = [64, 32, 16]
    output_channels = {
        'r': (2, 401),        # Single value output (no activation)
        'i': (2, 401),  # 10-class classification
    }
    
    # Create model
    model = MultiOutputTCN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_channels=output_channels,
    )
    # Sample input
    batch_size = 16
    input_data = torch.randn(batch_size, input_dim)
    # Forward pass
    outputs = model(input_data)
    
    
    for channel, output in outputs.items():
        print(f"{channel}: {output.shape}")
    
   

def test_Lout():
    L = 1
    # kernel_size = [32, 32, 32]
    # stride = [3, 3, 3]
    # padding = [0, 1, 0]
    # dilation = [1, 1, 1]
    # output_padding = [0, 1, 0]

    kernel_size = [32, 36, 32]
    stride = [4, 4, 4]
    padding = [0, 1, 0]
    dilation = [1, 1, 1]
    output_padding = [0, 1, 0]
    # output 401
    for i in range(len(kernel_size)):
        L = calculate_Lout(
            Lin=L,
            kernel_size=kernel_size[i],
            stride=stride[i],
            padding=padding[i],
            dilation=dilation[i],
            output_padding=output_padding[i]
        )
        print(f"Layer {i+1}: Lout = {L}")