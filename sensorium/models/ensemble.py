import torch
import torch.nn as nn

class EnsemblePrediction(nn.Module):
    """ Simple ensemble model that pools responses from models """
    
    def __init__(self, model_list, mode='mean'):
        super(EnsembleModel, self).__init__()
        
        self.model_list = nn.ModuleList( model_list )
        self.mode = mode
       
    def forward(self, *args, **kwargs):
        """ Forward function passes all arguments to individual models """
        
        # get responses of individual models
        Y_list = list()
        for model in self.model_list:
            Y_list.append( model(*args, **kwargs) )
        Y = torch.stack( Y_list, dim=0 )    
        
        # pool data depending on selected mode
        if self.mode == 'mean':
            Y = torch.mean( Y, dim=0 )
        elif self.mode == 'median':
            Y = torch.median( Y, dim=0 ).values   # ignore indicies
        elif self.mode == 'max':
            Y = torch.max( Y, dim=0 ).values  
        else:
            raise Exception('Unkown mode "{}"'.format(self.mode))

        return Y
