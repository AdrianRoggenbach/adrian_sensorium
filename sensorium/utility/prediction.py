""" Calculate and sort predictions of the model
Based on sensorium/utility/scores.py file

Adrian 2022-09-25 """

import os
import numpy as np
import torch
from tqdm import tqdm

from neuralpredictors.training import eval_state, device_state

def single_prediction_with_trial(model, dataloader, data_key, device="cuda"):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
        trial_ids: corresponding id based on values in data directory
    """

    target, output, trial_ids = torch.empty(0), torch.empty(0), torch.empty(0)
    for batch in dataloader:
        # batch is enumerated as list, first two are images and response
        
        images, responses = (
            batch[:2]
            if not isinstance(batch, dict)
            else (batch["inputs"], batch["targets"])
        )
        batch_kwargs = batch._asdict() if not isinstance(batch, dict) else batch
    
        # make sure trial_id is in the batch arguments
        if not 'trial_id' in batch_kwargs:
            raise Exception(
                "The trial_id is not batch. Add it with 'include_trial_id':True, in the "
                "dataset configuration"
            )
        ids = batch._asdict()['trial_id']
        
        with torch.no_grad():
            with device_state(model, device):
                output = torch.cat(
                    (
                        output,
                        (
                            model(images.to(device), data_key=data_key, **batch_kwargs)
                            .detach()
                            .cpu()
                        ),
                    ),
                    dim=0,
                )
            target = torch.cat((target, responses.detach().cpu()), dim=0)
            trial_ids = torch.cat((trial_ids, ids.detach().cpu()), dim=0)

    return target.numpy(), output.numpy(), trial_ids.numpy().flatten().astype(int)


def all_predictions_with_trial(model, dataloaders, device='cuda'):
    """
    Compute the predictions on all datasets and data splits
    
    Returns:
       results:  dictionary with keys: data_key,
                                           keys: splits
                                                  keys: 'target', 'output', 'trial_ids'
                                                  values: 2d arrays  (nr_trials, neurons/1)
    
    """
    
    results = dict()
    splits = list(dataloaders.keys())
    data_keys = list( dataloaders[splits[0]].keys() )
    
    for data_key in tqdm(data_keys, desc='Iterating datasets'):
        results[data_key] = dict()
        nr_trials = 0
        for split in splits:
            dl = dataloaders[split][data_key]
            
            target, output, trial_ids = single_prediction_with_trial(model, dl, data_key, device)
            results[data_key][split] = dict(target=target, output=output, trial_ids=trial_ids)
            nr_trials += len( trial_ids )
            if split == 'train':
                results[data_key]['nr_neurons'] = target.shape[1]
                
        results[data_key]['nr_trials'] = nr_trials
        
    return results
            
    
def merge_predictions(results):
    """ TODO """
    merged = dict()
    for data_key, data in results.items():
        target = np.zeros( (data['nr_trials'], data['nr_neurons']))
        output = np.zeros( (data['nr_trials'], data['nr_neurons']))
        trial_type = np.zeros( data['nr_trials'], dtype=int )
        
        type_conversion = dict(train=0, validation=1, test=2, final_test=3)
        
        # merge the data from the invidual data splits
        for split in type_conversion:
            split_dict = data[split]
            trials = split_dict['trial_ids']
            if len(trials) > 0:  # some splits might be empty like final_test
                target[trials,:] = split_dict['target']
                output[trials,:] = split_dict['output']
                trial_type[trials] = type_conversion[split]
            
        merged[data_key] = dict(target = target,
                                  output = output,
                                  trial_type = trial_type,
                                 )
    return merged



def sort_predictions_by_time(merged, dataset_sorting_path='notebooks/data/dataset_sortings.npy'):
    """ TODO """
    
    # dict with keys: data_key and values list with argsort for trials
    data_sorting = np.load(dataset_sorting_path, allow_pickle=True).item()
    
    sorted_results = dict()
    for data_key, data in merged.items():
        sorting = data_sorting[data_key]
        sorted_results[data_key] = dict()
        sorted_results[data_key]['target'] = data['target'][sorting,:]
        sorted_results[data_key]['output'] = data['output'][sorting,:]
        sorted_results[data_key]['trial_type'] = data['trial_type'][sorting]
        sorted_results[data_key]['trial_id'] = np.arange( len(sorting) )[sorting]
        
    return sorted_results



def inplace_add_behavior_to_sorted_predictions(sorted_results,
                                       dataset_sorting_path='notebooks/data/dataset_sortings.npy',
                                       data_folder = 'notebooks/data',
                                      ):
    """ TODO """
    
    # hardcoded transformation of data_key to folder name
    before = 'static'
    after = '-GrayImageNet-94c6ff995dac583098847cfecd43e7b6'

    for data_key in sorted_results:
        merged_folder = os.path.join( data_folder, before+data_key+after, 'merged_data' )
        sorting = np.load( os.path.join( merged_folder, 'sort_id.npy' ))
        behavior = np.load( os.path.join( merged_folder, 'behavior.npy' ))
        
        sorted_results[data_key]['pupil'] = behavior[sorting,0]
        sorted_results[data_key]['pupil_dt'] = behavior[sorting,1]
        sorted_results[data_key]['running'] = behavior[sorting,2]
        
        center = np.load( os.path.join( merged_folder, 'pupil_center.npy' ))
        sorted_results[data_key]['center'] = behavior[sorting,:]
        
    return   # change is done inplace in dictionary => no return value