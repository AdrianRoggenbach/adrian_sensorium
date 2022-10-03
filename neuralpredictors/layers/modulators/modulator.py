import torch
from torch import nn
from torch.nn import ModuleDict
import numpy as np


class HistoryGainModulator(nn.Module):
    def __init__(self, nr_neurons,
                 include_gain=False,
                 include_history=True,
                 nr_history=5,
                 per_neuron_gain_adjust=False,
                 behav_state=False,
                 nr_behav_state=10,
                 ):
        super().__init__()
        # save parameter
        self.include_gain = include_gain
        self.include_history = include_history
        self.per_neuron_gain_adjust = per_neuron_gain_adjust
        self.behav_state = behav_state
        
        if self.per_neuron_gain_adjust:
            # initialize all with 1
            self.gain_adjust = nn.Parameter( torch.ones(nr_neurons) )
        
        # initialize like linear layer, uniform between +-sqrt(1/nr)
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        max_val = np.sqrt( 1/nr_history )
        weights = torch.rand( (nr_neurons,nr_history) ) * (2*max_val) - max_val
        bias = torch.rand( nr_neurons ) * (2*max_val) - max_val
        self.history_weights = nn.parameter.Parameter( weights )
        self.history_bias = nn.parameter.Parameter( bias )
        
        if self.behav_state:
            # linear layer from hidden states to neurons
            self.state_encoder = nn.Linear( in_features=nr_behav_state,
                                            out_features=nr_neurons,
                                            bias=True )
            
            
    def forward(self, x, history, gain, state):
        # x: (batch, nr_neurons) Output of the encoding model which uses images+behavior
        # history: (batch, nr_neurons, nr_lags)
        # gain: (batch, 1)
        # state: (batch, nr_states)
        
        if self.include_history:
            # compute effect of history
            hist = torch.einsum( 'bnh,nh->bn', history, self.history_weights )
            hist = hist + self.history_bias
            x = x + hist    # add history and image response

        # compute reduced-rank prediction from last available timestep
        if self.behav_state:
            state_mod = self.state_encoder( state )
            x = x + state_mod
            
        # non-linearity
        x = nn.functional.elu(x) + 1
        
        if self.include_gain:
            # multiply response with a gain factor
            if self.per_neuron_gain_adjust:
                #           (batch, 1)   (batch,1)  (batch,nr_neurons)
                x = (1 + self.gain_adjust * gain)  *   x  
            else:
                x = (1 + gain) * x 
            
        return x
        
    def initialize(self, **kwargs):
        print('Initialize called but not implemented')
    
    def regularizer(self, data_key):
        return 0  #self[data_key].regularizer() * self.gamma_shifter
    
