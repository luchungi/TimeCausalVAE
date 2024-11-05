# Time-Causal VAE ⏱
The official code repository for paper [Time-Causal VAE: Robust Financial Time Series Generator](https://justinhou95.github.io/).

## Environments 🔩
* Python 3.11 and PyTorch 2.5

* Run the following commands to install python libraries:
  - `python -m pip install -r requirements/development.txt`
 
* You can also use `pip-tools` to regenerate the `requirements/development.txt` from `requirements/development.in` by the following command:
  - `python -m piptools compile requirements/train.in --output-file requirements/train.txt`
  - and then run `python -m pip install -r requirements/development.txt`

## Training 🧗🏻

You can find a example notebook [notebooks/example.ipynb](https://github.com/justinhou95/TimeCausalVAE/blob/main/notebooks/example.ipynb) for the training pipeline.

## Datasets and Trained Weight 📦

You can find the evaluation of trained models in the notebooks:

- [Black-Scholes](https://github.com/justinhou95/TimeCausalVAE/blob/main/notebooks/BlackScholes.ipynb)
- [Heston](https://github.com/justinhou95/TimeCausalVAE/blob/main/notebooks/Heston.ipynb)
- [Path dependent volatility model](https://github.com/justinhou95/TimeCausalVAE/blob/main/notebooks/PDV.ipynb)
- [S&P500 and VIX](https://github.com/justinhou95/TimeCausalVAE/blob/main/notebooks/SP500.ipynb)
- [Toy 2d examples](https://github.com/justinhou95/TimeCausalVAE/blob/main/notebooks/2Dtoydistributions.ipynb)
  
The trained models weights and training configurations are save in [trained_models](https://github.com/justinhou95/TimeCausalVAE/tree/main/trained_models)


![Image](https://github.com/justinhou95/TimeCausalVAE/blob/main/trained_models/Hestonprice_timestep_60/model_InfoCVAE_De_CLSTMRes_En_CLSTMRes_Prior_RealNVP_Con_Id_Dis_None_comment_None/InfoCVAE_training_2024-09-16_18-19-18/prices_real_fake.png)


## Contact
If you have any questions, please feel free to reach me out!




