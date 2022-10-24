<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

# SENSORIUM 2022 Competition

![plot](figures/Fig1.png)
SENSORIUM is a competition on predicting large scale mouse primary visual cortex activity. We will provide large scale datasets of neuronal activity in the visual cortex of mice. Participants will train models on pairs of natural stimuli and recorded neuronal responses, and submit the predicted responses to a set of test images for which responses are withheld.

Join our challenge and compete for the best neural predictive model!

For more information about the competition, vist our [website](https://sensorium2022.net/).

Have a look at our [White paper on arXiv](https://arxiv.org/abs/2206.08666), which describes the dataset and competition in detail.

# How to run the code

To reproduce the results of the last submission to the challenge (Model 4) run the notebooks in the folder "notebooks/submission_m4" in the order as indicated by the numbers.
These notebooks will generate new variables as regressors, fit 5 models with different seeds and test/val splits, and then create the submission file based on the ensemble of these 5 models.


# Short description of improvements to the model

The submitted model builds strongly on the model that was provided as a starting point. It uses the core module that learns to predict responses to natural images and behavioral variables (convolutional layers with image+behavior, readout location based on retinotopy, and shifter network). On top of that, a modulator network was implemented that uses information from the neural activity in the past (history, behavioral state, pupil independent gain) to explain larger parts of the single trial variability.

In detail, the following additions were made:
1. Normalize behavioral regressors more consistently across sessions (0 to 1 scaling instead of standard deviation)
2. Small changes to hyperparameters (number of kernels, retinotopy network, regularization)
3. History effects of neurons included (filter bank with different time scales for each neuron individually)
4. Behavioral state from the last timestep (based on reduced rank regression of next timepoint + non-negative matrix factorization)
5. Learned gain term that modulates the output (smoothly variying gain as parameter of the model)
6. Ensemble model to combine the mean of five independent models (different seed and test/val split)


