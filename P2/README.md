# Decision-Making-under-Uncertainty
This repo is for project assigned in the course Decision Making under Uncertainty Fall 2022.

In order to run the program, you need to install the required python packages. 

Run the following commands to install the requirements.
```
pip install -r requirements.txt
```

Once all the required packages are installed, run the following command to train
vanilla RL

```
python PPO.py False none
```

To train uncertainty aware RL with entropy (where reward = 1 - H), run

```
python PPO.py True entropy
```

To train uncertainty aware RL with dissonance (where reward = 1 - u_d), run

```
python PPO.py True dissonance
```

Once you have trained all the models, run the following command to generate the 
plots

```
python plots.py
```