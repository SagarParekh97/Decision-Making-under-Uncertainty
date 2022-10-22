# Decision-Making-under-Uncertainty
This repo is for project assigned in the course Decision Making under Uncertainty Fall 2022.

In order to run the program, you need to install the required python packages. 

Run the following commands to install the requirements.
```
pip install -r requirements.txt
```

Then install the package bnsobol from github.

```
git clone https://github.com/rballester/bnsobol.git
cd bnsobol
pip install .
```

Once all the required packages are installed, you can open the jupyter notebook to 
and run all the cells to produce the results. Since we use structure learning using greedy search which can get stuck at local maxima, you can get a different output BN for different runs of the program. The BN learned from the dataset that has been used in our report is saved as ```model_BN_good.bif```. When running the notebook, you will be prompted to either load the trained model or train a new model entirely.