# Weighting waiting: A decision-theoretic taxonomy of procrastination and delay

This repository contains code and data for the paper FILL_IN_THE_BLANK

Authors: [Sahiti Chebolu](https://www.kyb.tuebingen.mpg.de/person/107410/2549) and [Peter Dayan](https://www.mpg.de/12309357/biologische-kybernetik-dayan)

## Abstract
Why do today, what you can fail to do tomorrow? Pacing, postponing, and ultimately procrastinating tasks are widespread pathologies to which many succumb. Previous research has recognised multiple types of delay, influenced by a myriad of psychological and situational factors. However, the main mechanistic explanation, which concerns temporal discounting of delayed rewards, only encompasses a subset. Here, we introduce a systematic taxonomy of delay and procrastination within a reinforcement learning framework in which these choices are samples from policies associated with decision-making tasks. The mechanisms driving de lays are then closely tied to elements governing these policies, such as task structure and the (sub-)optimality of the solution strategy. Through a detailed analysis of real-world data on student engagement (Zhang & Ma, Scientific Reports, 2024), we illustrate how some of these diverse sources of delay can explain behavior. Our approach provides a theoretical foundation for understanding pacing and procrastination, enabling the integration of both established and novel mechanisms within a unified conceptual framework.

## Installation

1. clone repository 
   - for https: `git clone https://github.com/SahitiC/procrastination_taxonomy.git` or 
   - for ssh: `git clone git@github.com:SahitiC/procrastination_taxonomy.git`
2. create and activate python virtual environment: 
   - for python virtual environments:\
   create: `python3 -m venv .env`\
   activate: for macOS & Linux, `source .env/bin/activate` and for Windows, `.env\Scripts\activate`
   - for conda environments:\
   create: `conda create -n env`\
   activate: `conda activate env`
3. install packages in requirments.txt: 
   - for pip: \
   `pip install -r requirements.txt` 
   - for conda: \
   `conda config --add channels conda-forge` \
   `conda install --yes --file requirements.txt`

## Usage

**Please note that plots in each figure of the paper are generated separately by this code. model_fitting.py, recovery.py and recovery_fit_params.py are parallelised and take a long time to run. The outputs of these scripts are stored in the data/ folder and can be used directly.**
1. First, run the data pre-processing script on data from Zhang and Ma 2024:
   <code>
   python data_preprocessing.py
   </code>
2. Then, run the clustering code to reproduce Figure 2 and Supplementary Figure 8:
   <code>
   python data_clustering.py
   </code>
3. Simulate the models to reproduce Figures 4-5:
   <code>
   python simulations.py
   </code>
4. Fit models to data clusters:
   <code>
   python model_fitting.py
   </code>
5. Plot results of model fits and related metrics to reproduce Figures 6-7:
   <code>
   python model_fitting_plots.py
   </code>
6. Run model and parameter recovery with random and fitted parameters:
   <code>
   python recovery.py
   python recovery_fit_params.py
   </code>
7. Plot results of recovery analysis to reproduce Supplementary Figures 9-10:
   <code>
   python recovery_plots.py
   </code>

## Description

1. data/ - folder containing original data from Zhang and Ma 2024 and intermediate pre-processed, clustered data files; also contains output data from model fitting and recovery analyses
2. plots/ - folder containing all plots as vector images and final figures in the paper
The following modules contain some helper functions for further steps: 
3. mdp_algms.py - functions for algorithms that find the optimal policy in MDPs, based on dynamic programming
4. task_structure.py - functions for constructing reward/ effort functions based on various reward schedules and convex/ linear cost functions and also transition functions based on the transition structure in the different models
5. gen_data.py - functions for simulating trajectories from models
6. likelihoods.py - functions for calculating and maximising log likelihood (minimising negative log likelihood) of data under different models
7. constants.py - define few shared constants over all the models (states, actions, horizon, effort, shirk reward)
The following scripts implement various steps of the analyses:
8. data_preprocessing.py - remove unwanted columns, apply exclusion criteria, normalise data
9. data_clustering.py - apply k-means to cluster trajectories, plot trajectories in each cluster and distance matrix; order of plots not same as in paper
10. simulations.py - simulate six models that formalise various delay mechanisms, plot delays and completion rates with varying parameters
11. model_fitting.py - fit models to data using maximum likelihoood estimatation
12. model_fitting_plots.py - plot simulated trajectories from model fits, calculate model fit metrics
13. recovery.py - parameter and model recovery with randomly chosen parameters
14. recovery_fit_params.py - parameter and model recovery with fitted parameters
15. recovery_plots.py - plot results of recovery analysis
16. .gitignore - tell git to ignore some local files, please change this based on your local repo
17. requirements.txt - python packages required to run these files
18. CheboluDayan2024b.pdf - final pdf of the paper

## Citation

If you found this code or paper helpful, please cite us as: FILL_IN_THE_BLANK

## Contact

For any questions or comments, please contact us at <sahiti.chebolu@tuebingen.mpg.de>
