# SENSORIUM+ 2022 Competition

This repository contains the code that was used in the SENSORIUM+ competition (Website: https://sensorium2022.net/home) for the winning submission.

This repository is based on code from https://github.com/sinzlab/sensorium

# How to run the code

Run the following commands to clone the repository, set up a new anaconda environment and start a jupyter notebook:
```
git clone https://github.com/AdrianHoffmann/adrian_sensorium.git
cd adrian_sensorium
conda env create -f environment.yaml
conda activate adrian_sensorium
python -m ipykernel install --user --name env_sensorium
jupyter notebook
```

To reproduce the results of the last submission to the challenge (Model 4) run the notebooks in the folder "notebooks/submission_m4" in the order as indicated by the numbers. You might need to change the kernel to "env_sensorium" when starting the notebooks.
These notebooks will generate new variables as regressors, fit 5 models with different seeds and test/val splits, and then create the submission file based on the ensemble of these 5 models.

Alternatively, you can also run an updated version of the models by following the notebooks in https://github.com/AdrianHoffmann/adrian_sensorium/tree/main/notebooks/model_walkthrough

The code in the current version requires a computer with 32GB RAM and a GPU.


# Short description of improvements to the model

The submitted model builds strongly on the model that was provided as a starting point. It uses the core module that learns to predict responses to natural images and behavioral variables (convolutional layers with image+behavior, readout location based on retinotopy, and shifter network). On top of that, a modulator network was implemented that uses information from the neural activity in the past (history, behavioral state, pupil independent gain) to explain larger parts of the single trial variability.

In detail, the following additions were made:
1. Normalize behavioral regressors more consistently across sessions (0 to 1 scaling instead of standard deviation)
2. Small changes to hyperparameters (number of kernels, retinotopy network, regularization)
3. History effects of neurons included (filter bank with different time scales for each neuron individually)
4. Behavioral state from the last timestep (based on reduced rank regression of next timepoint + non-negative matrix factorization)
5. Learned gain term that modulates the output (smoothly variying gain as parameter of the model)
6. Ensemble model to combine the mean of five independent models (different seed and test/val split)


# Acknowledgements

This repository contains code from the following repositories:
- Sensorium baseline model+repository (forked): https://github.com/sinzlab/sensorium
- Model building blocks from the Sinz lab: https://github.com/sinzlab/neuralpredictors/tree/v0.3.0
- Rastermap core code from Carsen Stringer and Marius Pachitariu: https://github.com/MouseLand/rastermap

We acknowledge the use of Fenix Infrastructure resources, which are partially funded from the European Union’s Horizon 2020 research and innovation programme through the ICEI project under the grant agreement No. 800858
