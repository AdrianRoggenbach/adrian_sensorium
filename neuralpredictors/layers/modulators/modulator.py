import torch
from torch import nn
from torch.nn import ModuleDict
import numpy as np


class HistoryGainModulator(nn.Module):
    def __init__(self, nr_neurons, nr_history=5,
                 per_neuron_gain_adjust=False,
                 behav_state=False,
                 nr_hidden_behav_state=10,
                 ):
        super().__init__()
        # save parameter
        self.per_neuron_gain_adjust = per_neuron_gain_adjust
        self.behav_state = behav_state
        self.nr_hidden_behav_state = nr_hidden_behav_state
        
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
            # reduced-rank matrix (nr_neurons, nr_hidden_behav_state)
            max_val = np.sqrt( 1/nr_neurons )
            weights = torch.rand( (nr_neurons,nr_hidden_behav_state) ) * (2*max_val) - max_val  
            self.reduced_rank_weights = nn.parameter.Parameter( weights )
            
            # readout matrix indiviual for each neuron (nr_hidden_behav_state, nr_neurons)
            max_val = np.sqrt( 1/nr_hidden_behav_state )
            weights = torch.rand( (nr_hidden_behav_state, nr_neurons) ) * (2*max_val) - max_val 
            bias = torch.rand( nr_neurons ) * (2*max_val) - max_val
            self.rr_readout_weights = nn.parameter.Parameter( weights )
            self.rr_readout_bias = nn.parameter.Parameter( bias )
            
            
    def forward(self, x, history, gain):
        # x: (batch, nr_neurons) Output of the encoding model which uses images+behavior
        # history: (batch, nr_neurons, nr_lags)
        # gain: (batch, 1)
        
        # compute effect of history
        hist = torch.einsum( 'bnh,nh->bn', history, self.history_weights )
        hist = hist + self.history_bias
        x = x + hist    # add history and image response
        
        # compute reduced-rank prediction from last available timestep
        if self.behav_state:
            last_response = history[:,:,0]   # first entry is lag -1
            reduced_rank = torch.einsum( 'bn,nh->bh', last_response, self.reduced_rank_weights)
            per_neuron_output = torch.einsum( 'bh,hn->bn', reduced_rank, self.rr_readout_weights)
            # per_neuron_output = per_neuron_output + self.rr_readout_bias
            x = x + per_neuron_output
            
        # non-linearity
        x = nn.functional.elu(x) + 1
        
        # multiply response with a gain factor
        # if self.per_neuron_gain_adjust:
        #     #           (batch, 1)   (batch,1)  (batch,nr_neurons)
        #     x = (1 + self.gain_adjust * gain)  *   x  
        # else:
        #     x = (1 + gain) * x 
            
        return x
        
    def initialize(self, **kwargs):
        print('Initialize called but not implemented')
    
    def regularizer(self, data_key):
        return 0  #self[data_key].regularizer() * self.gamma_shifter
    
