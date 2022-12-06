# Decision-Making-under-Uncertainty
This repo is for project assigned in the course Decision Making under Uncertainty Fall 2022.

In order to run the program, you need to install the required python packages. 

Run the following commands to install the requirements.
```
pip install -r requirements.txt
```

Once all the required packages are installed, do the following to install the custom gym environment

```
cd gym_grid
pip install -e .
cd ..
```

Run the following command to train
vanilla RL

```
python PPO.py false none none
```

To train uncertainty aware RL with entropy (where reward = 1 - H), run

```
python PPO.py true false H
```

To train uncertainty aware RL with dissonance (where reward = 1 - D), run

```
python PPO.py true false D
```

To train uncertainty aware RL with maximized vacuity (where reward = 1 - V), run

```
python PPO.py true false V
```

To train uncertainty aware RL with entropy (where exploration coefficient e = e(1 - H)), run

```
python PPO.py true true H
```

To train uncertainty aware RL with dissonance (where exploration coefficient e = e(1 - D)), run

```
python PPO.py true true D
```

To train uncertainty aware RL with maximized uncertainty (where exploration coefficient e = e(1 - V)), run

```
python PPO.py true true V
```

Once you have trained all the models, run the following command to generate the 
plots

```
python plots.py
```