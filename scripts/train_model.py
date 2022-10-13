""" Script to train a sensorium model

Adrian 2022-10-03 """

print('Running script train_model.py')

import sys, os
os.chdir('..')
print('Working directory:', os.getcwd() )
sys.path.insert(0, '.')  # hacky solution for now, TODO: fix

# imports
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import shutil

import warnings
warnings.filterwarnings('ignore')
from nnfabrik.builder import get_data, get_model, get_trainer
from sensorium.utility.training import read_config, print_t, set_seed

# read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', default='model_for_testing')
args = parser.parse_args()

model_name = args.model_name

save_folder = os.path.join( 'saved_models', model_name )
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
    
config_file = os.path.join( 'saved_models', '02_jobs', model_name+'.yaml' )
shutil.copy( config_file, os.path.join(save_folder, 'config.yaml' ))
config = read_config( config_file )

if config['verbose'] > 0:
    print_t('Loading data for "{}"'.format(model_name))

set_seed( config['model_seed'] )  # seed all random generators

#####################################
## DATALOADER
######################################
if config['data_sets'][0] == 'all':
    basepath = "notebooks/data/"
    filenames = [os.path.join(basepath, file) for file in os.listdir(basepath) if ".zip" in file ]
    filenames = [file for file in filenames if 'static26872-17-20' not in file]
else:
    filenames = config['data_sets']
    # filenames like ['notebooks/data/static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip', ]

dataset_fn = config['dataset_fn']  # 'sensorium.datasets.static_loaders'
dataset_config = {'paths': filenames,
                  **config['dataset_config'],
                 }

dataloaders = get_data(dataset_fn, dataset_config)

#####################################
## MODEL
######################################
# Instantiate model
model_fn = config['model_fn']     # e.g. 'sensorium.models.modulated_stacked_core_full_gauss_readout'
model_config = config['model_config']

model = get_model(model_fn=model_fn,
                  model_config=model_config,
                  dataloaders=dataloaders,
                  seed=config['model_seed'],
                 )

#####################################
## LOAD PRETRAINED CORE
######################################
if config['use_pretrained_core']:
    save_file = config['pretrained_model_file']
    pretrained_state_dict = torch.load(save_file)
    
    # keep only values of core
    core_only = {k:v for k, v in pretrained_state_dict.items() if 'core.' in k}

    # set only pretrained core values
    ret = model.load_state_dict(core_only, strict=False)


#####################################
## TRAINER
######################################
trainer_fn = config['trainer_fn']   # "sensorium.training.standard_trainer"
trainer_config = config['trainer_config']

trainer = get_trainer(trainer_fn=trainer_fn, 
                     trainer_config=trainer_config)


#####################################
## TRAINING AND EVALUATION
######################################
print_t('Start of model training')
# Train model
validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)
print_t('Model training finished')
            
# Save model
torch.save(model.state_dict(), os.path.join(save_folder, 'saved_model_v1.pth') )


# Run predictions
from sensorium.utility import prediction

# calculate predictions per dataloader
results = prediction.all_predictions_with_trial(model, dataloaders)

# merge predictions, sort in time and add behavioral variables
merged = prediction.merge_predictions(results)
sorted_res = prediction.sort_predictions_by_time(merged)
prediction.inplace_add_behavior_to_sorted_predictions(sorted_res)

# calculate correlations on splits
dataframe_entries = list()
trial_trans = { 0:'Train', 1:'Val', 2:'Test', 3:'Final Test'}
keys = list( sorted_res.keys() )

for key in keys:
    # calculate correlations
    ses_data = sorted_res[key]
    nr_neurons = ses_data['output'].shape[1]
    trial_type = ses_data['trial_type']

    for i in range(nr_neurons):
        true = ses_data['target'][:,i]
        pred = ses_data['output'][:,i]

        for split in range(3):
            cor = np.corrcoef( true[ trial_type==split ], pred[ trial_type==split ])[1,0]
            if np.isnan(cor):
                cor=0

            dataframe_entries.append(
                        dict(model=model_name, key=key, neuron=i,
                             split=trial_trans[split], cor=cor)
                        )

df = pd.DataFrame( dataframe_entries )

if config['save_csv']:
    # save DataFrame as csv
    path = os.path.join( 'saved_models', '00_csv_results', model_name+'.csv' )
    df.to_csv(path)

if config['save_predictions_npy']:
    # save also the predictions for each neuron
    np.save( os.path.join(save_folder, model_name+'.npy'), sorted_res)

print_t('Done with evaluation. Exiting...')
