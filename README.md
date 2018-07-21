# Dropout Feature Ranking for Deep Learning Model

[ArXiv](https://arxiv.org/abs/1712.08645) | [BibTex](#citing)

This is the code for reproducing the paper result. We propose a new method to gain feature importance from deep neural network.

## Run
0. Requirements:
    * Install python3
    * Install package pandas, seaborn, glmnet_python
    * Install pytorch 0.3 version (It does not support 0.4)
    * If you want to replicate the results of real datasets, download the datasets and put it under folder exp/DFRdatasets/data/.
        * [Support2](https://drive.google.com/drive/folders/1oStZbCqy_nKNW_iG8f_UlR_l0SR_NcVZ?usp=sharing)
        * [MiniBooNE](https://drive.google.com/drive/folders/1Kq-pyRsSMwi0hVMuN3X305g5BtO-hC5n?usp=sharing)
        * [Online News Popularity](https://drive.google.com/drive/folders/17WSvU9Y_uMjPgnd5B8hf_AoLXxhArBeH?usp=sharing)
        * [Year Prediction MSD](https://drive.google.com/drive/folders/1liZTxJNttYeNvTzIWAJIlqLEzQYAPjYa?usp=sharing)
    
 1. Running examples:
    * Running simulation
    ```bash
    python mlp_predict.py --dataset InteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank enet_rank lasso_rank
    python mlp_predict.py --dataset NoInteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank enet_rank lasso_rank

    ```
    * Support2 with zeroing out the feature
    ```bash
    python mlp_predict.py --dataset support2 --identifier 0111 --rank_func all_rank --test_func nn_test_zero
    ```
    * MiniBooNE
    ```bash
    python mlp_predict.py --dataset MiniBooNE --identifier 0111 \
    --rank_func nn_rank:0.1 nn_rank:0.01 marginal_rank rf_rank zero_rank shuffle_rank \
    random_rank dfs_rank enet_rank lasso_rank --test_func nn_test_zero
    ```
    * Year MSD
    ```bash
    python mlp_predict.py --dataset YearMSD --identifier 0111 \
    --rank_func nn_rank:1 marginal_rank rf_rank zero_rank shuffle_rank \
    random_rank dfs_rank enet_rank lasso_rank nn_rank:0.1  --test_func nn_test_retrain
    ```
    See notebooks for the further analysis and reproducing figures
    
2. Further questions?
    * Post in the issue or send to my email kingsley@cs.toronto.edu

## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.

## Citing
```
@article{chang2017dropout,
  title={Dropout Feature Ranking for Deep Learning Models},
  author={Chang, Chun-Hao and Rampasek, Ladislav and Goldenberg, Anna},
  journal={arXiv preprint arXiv:1712.08645},
  year={2017}
}
```
