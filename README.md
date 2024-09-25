# Weighting waiting: A decision-theoretic taxonomy of procrastination and delay

This repository contains code and data for 

Authors: [Sahiti Chebolu](https://www.kyb.tuebingen.mpg.de/person/107410/2549) and [Peter Dayan](https://www.mpg.de/12309357/biologische-kybernetik-dayan)

## Abstract
Why do today, what you can fail to do tomorrow? Pacing, postponing, and ultimately procrastinating tasks are widespread pathologies to which many succumb. Previous research has recognised multiple types of delay, influenced by a myriad of psychological and situational factors. However, the main mechanistic explanation, which concerns temporal discounting of delayed rewards, only encompasses a subset. Here, we introduce a systematic taxonomy of delay and procrastination within a reinforcement learning framework in which these choices are samples from policies associated with decision-making tasks. The mechanisms driving de lays are then closely tied to elements governing these policies, such as task structure and the (sub-)optimality of the solution strategy. Through a detailed analysis of real-world data on student engagement (Zhang & Ma, Scientific Reports, 2024), we illustrate how some of these diverse sources of delay can explain behavior. Our approach provides a theoretical foundation for understanding pacing and procrastination, enabling the integration of both established and novel mechanisms within a unified conceptual framework.

## Installation

1. clone repository \
   for https: `git clone https://github.com/SahitiC/procrastination_taxonomy.git` or \
   for ssh: `git clone git@github.com:SahitiC/procrastination_taxonomy.git`
2. create and activate python virtual environment using your favourite environment manager (pip, conda etc)
3. install packages in requirments.txt: \
   for pip: \
   `pip install -r requirements.txt` \
   for conda: \
   `conda config --add channels conda-forge` \
   `conda install --yes --file requirements.txt
