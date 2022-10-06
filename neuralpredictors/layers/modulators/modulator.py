import torch
from torch import nn
from torch.nn import ModuleDict
import numpy as np
from scipy import signal

class HistoryGainModulator(nn.Module):
    def __init__(self, nr_neurons,
                 include_gain=False,
                 include_history=True,
                 nr_history=5,
                 per_neuron_gain_adjust=False,
                 behav_state=False,
                 nr_behav_state=10,
                 behav_alpha=0,
                 ):
        super().__init__()
        # save parameter
        self.include_gain = include_gain
        self.include_history = include_history
        self.per_neuron_gain_adjust = per_neuron_gain_adjust
        self.behav_state = behav_state
        self.behav_alpha = behav_alpha
        
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
            
            
    def forward(self, x, history, gain, state, sort_id):
        # x: (batch, nr_neurons) Output of the encoding model which uses images+behavior
        # history: (batch, nr_neurons, nr_lags)
        # gain: (batch, 1)
        # state: (batch, nr_states)
        # sort_id is ignored (only used in HistoryOwnGainModulator, see below)
        
        
        if self.include_history:
            # compute effect of history
            hist = torch.einsum( 'bnh,nh->bn', history, self.history_weights )
            hist = hist + self.history_bias
            x = x + hist    # add history
            
        # add additional signal based on the behavioral state
        if self.behav_state:
            state_mod = self.state_encoder( state )
            x = x + nn.functional.elu( state_mod )
            
        # non-linearity for final output of stimulus segment
        x = nn.functional.elu(x) + 1
        
        # modify stimulus response with gain
        if self.include_gain:
            # multiply response with a gain factor
            if self.per_neuron_gain_adjust:
                #       (nr_neurons, 1)   (batch,1)  (batch,nr_neurons)
                x = (1 + self.gain_adjust * gain)  *   x  
            else:
                x = (1 + gain) * x 

        return x
        
    def initialize(self, **kwargs):
        print('Initialize called but not implemented')
    

    def regularizer(self):
        if self.behav_state and (self.behav_alpha > 0):
            # L1 regularization
            abs_weight_sum = 0
            for p in self.state_encoder.parameters():
                abs_weight_sum += p.abs().sum()
            return abs_weight_sum * self.behav_alpha
        
        else:
            return 0  #self[data_key].regularizer() * self.gamma_shifter
        
        
        
        
class HistoryOwnGainModulator(nn.Module):
    def __init__(self, nr_neurons,
                 nr_trials,
                 include_gain=False,
                 include_history=True,
                 nr_history=5,
                 per_neuron_gain_adjust=False,
                 behav_state=False,
                 nr_behav_state=10,
                 behav_alpha=0,
                 gain_kernel_std=30,
                 diff_reg=1000,
                 ):
        super().__init__()
        # save parameter
        self.nr_trials = nr_trials
        self.include_gain = include_gain
        self.include_history = include_history
        self.per_neuron_gain_adjust = per_neuron_gain_adjust
        self.behav_state = behav_state
        self.behav_alpha = behav_alpha
        self.diff_reg = diff_reg
        
        if self.include_gain:
            self.own_gain = nn.Parameter( torch.zeros(nr_trials) )
            # kernel is half a gaussian, to keep causal smoothing
            window = signal.gaussian(201, std=gain_kernel_std)
            window[0:100] = 0
            self.gain_kernel = torch.zeros( (1,1,201) )
            self.gain_kernel[0,0,:] = torch.Tensor(window)
            # self.gain_kernel[0,0,:] = torch.tensor( [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05] )
            
            
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
            
            
    def forward(self, x, history, gain, state, sort_id, device='cuda'):
        # x: (batch, nr_neurons) Output of the encoding model which uses images+behavior
        # history: (batch, nr_neurons, nr_lags)
        # gain: (batch, 1)
        # state: (batch, nr_states)
        # sort_id: (batch, 1)
        
        
        if self.include_history:
            # compute effect of history
            hist = torch.einsum( 'bnh,nh->bn', history, self.history_weights )
            hist = hist + self.history_bias
            x = x + hist    # add history
            
        # add additional signal based on the behavioral state
        if self.behav_state:
            state_mod = self.state_encoder( state )
            x = x + nn.functional.elu( state_mod )
            
        # non-linearity for final output of stimulus segment
        x = nn.functional.elu(x) + 1
        
        # modify stimulus response with gain
        if self.include_gain:
            # transform sort_ids into one-hot encoding
            nr_batch = x.shape[0]
            one_hot = torch.zeros( (nr_batch, 1, self.nr_trials)).to(device)
            for i, s_id in enumerate( sort_id[:,0] ):
                one_hot[i,0,int(s_id)] = 1
                
            # smooth one_hot encoding along trial dimension
            # one_hot has shape (batch, 1, nr_trials), interpreted as 1 channel
            # kernel has shape (1, 1, length), one input and one output channel
            self.gain_kernel.to(device)
            smooth = nn.functional.conv1d(one_hot, self.gain_kernel, padding='same')
            
            # scalar product between smooth and saved gain vector (along trials)
            batch_gain = torch.einsum( 'bet,t->be', smooth, self.own_gain )
            
            # transform onto positive values only (0 mapped to 1)
            batch_gain = nn.functional.elu( batch_gain ) + 1
            
            if self.per_neuron_gain_adjust:
                #       (nr_neurons)       (batch,1)  (batch,nr_neurons)
                x = (1 + self.gain_adjust * batch_gain)  *   x  
            else:
                x = batch_gain * x 

        return x
        
    def initialize(self, **kwargs):
        print('Initialize called but not implemented')
    

    def regularizer(self):
        if self.include_gain and (self.diff_reg > 0):
            # L2 regularization of difference of gains

            return self.own_gain.diff().square().sum() * self.diff_reg
        
        else:
            return 0 
        
        
