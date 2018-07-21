#!/usr/bin/env bash

python mlp_predict.py --dataset wineqaulity --identifier 0105-r1 \
--rank_func nn_rank --reg_coef 1. --test_func svm_rbf_test && python mlp_predict.py --dataset wineqaulity --identifier 0105-r0.1 \
--rank_func nn_rank --reg_coef 0.1 --test_func svm_rbf_test && python mlp_predict.py --dataset wineqaulity --identifier 0105-r0.01 \
--rank_func nn_rank --reg_coef 0.01 --test_func svm_rbf_test

python mlp_predict.py --dataset wineqaulity --identifier 0105 \
--rank_func rf_rank --reg_coef 0.1 --test_func joint_test && python mlp_predict.py --dataset wineqaulity --identifier 0105-r0.01 \
--rank_func nn_rank --reg_coef 0.01 --test_func joint_test

python mlp_predict.py --dataset support2 --identifier 0105 \
--rank_func rf_rank --reg_coef 0.1 --test_func joint_test && python mlp_predict.py --dataset support2 --identifier 0105-r0.1 \
--rank_func nn_rank --reg_coef 0.1 --test_func joint_test && python mlp_predict.py --dataset support2 --identifier 0105-r0.01 \
--rank_func nn_rank --reg_coef 0.01 --test_func joint_test && python mlp_predict.py --dataset support2 --identifier 0105-r0.001 \
--rank_func nn_rank --reg_coef 0.001 --test_func joint_test && python mlp_predict.py --dataset wineqaulity --identifier 0105-r0.001 \
--rank_func nn_rank --reg_coef 0.001 --test_func joint_test

python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0105 \
--rank_func rf_rank --reg_coef 0.1 --test_func joint_test && python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0105-r0.1 \
--rank_func nn_rank --reg_coef 0.1 --test_func joint_test && python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0105-r0.01 \
--rank_func nn_rank --reg_coef 0.01 --test_func joint_test && python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0105-r0.001 \
--rank_func nn_rank --reg_coef 0.001 --test_func joint_test

# New classification for online news!
python mlp_predict.py --dataset ClassificationONPLoader --identifier 0105 \
--rank_func rf_rank --reg_coef 0.1 --test_func joint_test && python mlp_predict.py --dataset ClassificationONPLoader --identifier 0105-r0.005 \
--rank_func nn_rank --reg_coef 0.005 --test_func joint_test && python mlp_predict.py --dataset ClassificationONPLoader --identifier 0105 \
--rank_func marginal_rank --reg_coef 0.005 --test_func joint_test

# New support2 with rf / nn0.001
python mlp_predict.py --dataset support2 --identifier 0105 \
--rank_func rf_rank --reg_coef 0.1 --test_func joint_test && python mlp_predict.py --dataset support2 --identifier 0105-r0.001 \
--rank_func nn_rank --reg_coef 0.001 --test_func joint_test

# New support2 with prediction of slos!
python mlp_predict.py --dataset RegSupport2Loader --identifier 0105 \
--rank_func rf_rank --reg_coef 0.1 --test_func joint_test && python mlp_predict.py --dataset RegSupport2Loader --identifier 0105-r0.001 \
--rank_func nn_rank --reg_coef 0.001 --test_func joint_test

python mlp_predict.py --dataset support2 --identifier 0105-r0.01 \
--rank_func nn_rank zero_rank shuffle_rank --reg_coef 0.01 --test_func joint_test


python mlp_predict.py --dataset wineqaulity --identifier 0105 \
--rank_func zero_rank shuffle_rank random_rank --reg_coef 0.01 --test_func joint_test

python mlp_predict.py --dataset ClassificationONPLoader --identifier 0105-r0.01 \
--rank_func nn_rank zero_rank shuffle_rank --reg_coef 0.01 --test_func joint_test

python mlp_predict.py --dataset support2 --identifier 0105-r0.01 \
--rank_func random_rank --reg_coef 0.01 --test_func joint_test

python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0105-r0.01 \
--rank_func nn_rank rf_rank zero_rank random_rank --reg_coef 0.01 --test_func joint_test

python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0105-r0.01 \
--rank_func nn_rank rf_rank zero_rank random_rank --reg_coef 0.01 --test_func joint_test

python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0105-r0.01 \
--rank_func nn_rank rf_rank zero_rank random_rank --reg_coef 0.01 --test_func joint_test

# test these 2
python mlp_predict.py --dataset wineqaulity --identifier 0105 \
--rank_func enet_rank lasso_rank --test_func joint_test

# Test everything in NN classifier with 3 datasets: wineqaulity (reg), support2 (cls),
# classificationONPLoader (cls)
python mlp_predict.py --dataset wineqaulity --identifier 0111 \
--rank_func all_rank --test_func nn_test_zero && python mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func all_rank --test_func nn_test_zero && python mlp_predict.py --dataset ClassificationONPLoader --identifier 0111 \
--rank_func all_rank --test_func nn_test_zero && python mlp_predict.py --dataset RegSupport2Loader --identifier 0111 \
--rank_func all_rank --test_func nn_test_zero

python mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func all_rank --test_func nn_test_retrain --gpu-ids 2

python mlp_predict.py --dataset wineqaulity --identifier 0111 \
--rank_func all_rank --test_func nn_test_retrain

python mlp_predict.py --dataset ClassificationONPLoader --identifier 0111 \
--rank_func all_rank --test_func nn_test_retrain

python mlp_predict.py --dataset wineqaulity --identifier 0111 \
--rank_func nn_rank:0.001 --test_func nn_test_zero

python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0111 \
--rank_func all_rank --test_func nn_test_zero

python mlp_predict.py --dataset wineqaulity --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.001 nn_rank:0.005 --test_func nn_test_zero

python mlp_predict.py --dataset wineqaulity --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.001 nn_rank:0.005 --test_func nn_test_zero

python mlp_predict.py --dataset YearMSD --identifier 0111 \
--rank_func nn_rank:1 nn_rank:0.1 --test_func nn_test_zero

python mlp_predict.py --dataset YearMSD --identifier 0111 \
--rank_func marginal_rank rf_rank zero_rank shuffle_rank \
random_rank dfs_rank enet_rank lasso_rank --test_func nn_test_zero

#0113
python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0111 \
--rank_func nn_rank --test_func nn_test_zero --gpu-ids 2


python test_nn_hyperparam.py --dataset OnlineNewsPopularity --gpu-ids 2

python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0111 \
--rank_func marginal_rank rf_rank zero_rank shuffle_rank \
random_rank dfs_rank enet_rank lasso_rank --test_func nn_test_zero

python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0111 \
--rank_func iter_zero_rank iter_shuffle_rank --test_func nn_test_zero && python mlp_predict.py --dataset ClassificationONPLoader --identifier 0111 \
--rank_func iter_zero_rank iter_shuffle_rank --test_func nn_test_zero && python mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func iter_zero_rank iter_shuffle_rank --test_func nn_test_zero && python mlp_predict.py --dataset YearMSD --identifier 0111 \
--rank_func iter_zero_rank iter_shuffle_rank --test_func nn_test_zero

python mlp_predict.py --dataset OnlineNewsPopularity --identifier 0111 \
--rank_func all_rank --test_func nn_test_retrain

python mlp_predict.py --dataset MIMIC --identifier 0111 \
--rank_func all_rank --test_func nn_test_zero

python mlp_predict.py --dataset MIMIC --identifier 0111 \
--rank_func all_rank --test_func nn_test_retrain

# pure cpu
python mlp_predict.py --dataset MIMIC --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.01 random_rank rf_rank zero_rank marginal_rank shuffle_rank \
dfs_rank enet_rank lasso_rank --test_func nn_test_zero

python mlp_predict.py --dataset MIMIC --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.01 random_rank rf_rank zero_rank marginal_rank shuffle_rank \
dfs_rank enet_rank lasso_rank --test_func nn_test_retrain

python mlp_predict.py --dataset MiniBooNE --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.01 marginal_rank rf_rank zero_rank shuffle_rank \
random_rank dfs_rank enet_rank lasso_rank --test_func nn_test_zero

python mlp_predict.py --dataset MiniBooNE --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.01 marginal_rank rf_rank zero_rank shuffle_rank \
random_rank dfs_rank enet_rank lasso_rank --test_func nn_test_retrain

python mlp_predict.py --dataset MIMIC --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.01 --test_func nn_test_zero

python mlp_predict.py --dataset MiniBooNE --identifier 0111 \
--rank_func lasso_rank enet_rank dfs_rank random_rank shuffle_rank zero_rank  \
--test_func nn_test_retrain

python mlp_predict.py --dataset YearMSD --identifier 0111 \
--rank_func nn_rank:1 marginal_rank rf_rank zero_rank shuffle_rank \
random_rank dfs_rank enet_rank lasso_rank nn_rank:0.1  --test_func nn_test_retrain

# Rerun MIMIC with validation loader
python mlp_predict.py --dataset MIMIC --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.01 --test_func nn_test_zero

# Test with different seeds!!
python -u mlp_predict.py --dataset support2 --identifier 0111-s36 --seed 36 \
--rank_func nn_rank --test_func nn_test_zero && python -u mlp_predict.py --dataset support2 --identifier 0111-s48 --seed 48 \
--rank_func nn_rank --test_func nn_test_zero

python -u mlp_predict.py --dataset HIGGS --identifier 0111 \
--rank_func nn_rank:0.1 nn_rank:0.01 random_rank rf_rank \
zero_rank marginal_rank shuffle_rank \
dfs_rank enet_rank lasso_rank --test_func nn_test_zero > logs/0121-HIGGS-zero.log

python -u mlp_predict.py --dataset FMA --identifier 0111 \
--rank_func rf_rank \
zero_rank marginal_rank shuffle_rank \
dfs_rank enet_rank lasso_rank --test_func nn_test_zero > logs/0121-FMA-zero.log &

# Add dropout & batchnorm for MIMIC / and early stopping. Do again!
python mlp_predict.py --dataset MIMIC --identifier 0111 \
--rank_func nn_rank:0.01 marginal_rank rf_rank zero_rank nn_rank:0.1 shuffle_rank \
random_rank dfs_rank enet_rank lasso_rank --test_func nn_test_zero

#run on gpu (not finished)
python -u mlp_predict.py --dataset HIGGS --identifier 0111 --rank_func nn_rank:0.001 nn_rank:0.01 --test_func nn_test_zero > logs/0123-HIGGS-zero.log &

#tune the hyperparams to reduce running time. Rerun
python -u mlp_predict.py --dataset HIGGS --identifier 0111 --rank_func random_rank nn_rank:0.001 nn_rank:0.01 --test_func nn_test_zero > logs/0124-HIGGS-zero.log &

# cpu have not run
python -u mlp_predict.py --dataset HIGGS --identifier 0111 --rank_func rf_rank zero_rank marginal_rank shuffle_rank dfs_rank enet_rank lasso_rank --test_func nn_test_zero > logs/0123-HIGGS-cpu-zero.log &

python -u mlp_predict.py --dataset MIMIC --identifier 0111 --rank_func nn_rank:0.0001 nn_rank:0.00001 --test_func nn_test_zero > logs/0123-mimic-val.log &

# Randomize
python -u mlp_predict.py --dataset support2 --identifier 0111-s36 --seed 36 \
--rank_func nn_rank:0.01 --test_func nn_test_zero && python -u mlp_predict.py --dataset support2 --identifier 0111-s48 --seed 48 \
--rank_func nn_rank:0.01 --test_func nn_test_zero  && python -u mlp_predict.py --dataset support2 --identifier 0111-s12 --seed 12 \
--rank_func nn_rank:0.01 --test_func nn_test_zero && python -u mlp_predict.py --dataset support2 --identifier 0111-s24 --seed 24 \
--rank_func nn_rank:0.01 --test_func nn_test_zero

python -u mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func nn_rank:0.1 --test_func nn_test_zero && python -u mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func nn_rank:1 --test_func nn_test_zero && python -u mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func nn_rank:0.001 --test_func nn_test_zero && python -u mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func nn_rank:0.0001 --test_func nn_test_zero && python -u mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func nn_rank:0.01 --test_func nn_test_zero

python -u mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func nn_rank:0 --test_func nn_test_zero

python -u mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func nn_middle_rank:0.001 nn_middle_rank:1e-4 --test_func nn_test_zero

python -u mlp_predict.py --dataset MIMIC --identifier 0111 --rank_func nn_middle_rank:1e-4 nn_middle_rank:1e-5 nn_middle_rank:1e-3 --test_func nn_test_zero > logs/0124-mimic-middle.log &

python -u mlp_predict.py --dataset CreditCard --identifier 0111 --rank_func nn_rank:0.001 nn_rank:0.01 marginal_rank nn_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank enet_rank lasso_rank --test_func nn_test_zero > logs/0124-creditcard-zero.log &

#Fuck!!!!! They close all my sessions!!!!
python -u mlp_predict.py --dataset support2 --identifier 0111 \
--rank_func nn_middle_rank:0.001 nn_middle_rank:1e-4 --test_func nn_test_zero

python -u mlp_predict.py --dataset YearMSD --identifier 0111 \
--rank_func zero_rank shuffle_rank \
random_rank dfs_rank enet_rank lasso_rank nn_rank:0.1  --test_func nn_test_retrain

python -u mlp_predict.py --dataset FMA --identifier 0111 \
--rank_func rf_rank \
zero_rank marginal_rank shuffle_rank \
dfs_rank enet_rank lasso_rank --test_func nn_test_zero > logs/0125-FMA-zero.log &

python mlp_predict.py --dataset InteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank enet_rank lasso_rank

python mlp_predict.py --dataset NoInteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank enet_rank lasso_rank

python -u mlp_predict.py --dataset InteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank:0.5 enet_rank lasso_rank --test_func nn_test_zero > logs/0127_inter_zero.log && python -u mlp_predict.py --dataset InteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank:0.5 enet_rank lasso_rank --test_func nn_test_retrain > logs/0127_inter_retrain.log

python -u mlp_predict.py --dataset NoInteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank:0.5 enet_rank lasso_rank --test_func nn_test_zero > logs/0127_nointer_zero.log && python -u mlp_predict.py --dataset NoInteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank:0.5 enet_rank lasso_rank --test_func nn_test_retrain > logs/0127_nointer_retrain.log

python mlp_predict.py --dataset YearMSD --identifier 0111 \
--rank_func nn_rank:1 marginal_rank rf_rank zero_rank shuffle_rank \
random_rank dfs_rank enet_rank lasso_rank nn_rank:0.1  --test_func nn_test_retrain

python -u mlp_predict.py --dataset MoreInteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank:0.1 enet_rank lasso_rank --test_func nn_test_zero > logs/0128_moreinter_zero.log
python -u mlp_predict.py --dataset MoreInteractionSimulation --rank_func nn_rank:0.1 nn_rank:0.5 nn_rank:1 nn_rank:0.05 marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank:0.1 enet_rank lasso_rank --test_func nn_test_retrain > logs/0129_moreinter_retrain.log

python -u mlp_predict.py --dataset InteractionSimulation --rank_func nn_middle_rank:0.1 nn_middle_rank:0.01 nn_middle_rank:0.001 nn_middle_rank:1 nn_middle_rank:0.5 --test_func nn_test_zero > logs/0127_inter_middle.log

# weaker part??

python -u mlp_predict.py --dataset CorrelatedInteractionSimulation --rank_func nn_rank:0.005 nn_rank:0.05 nn_rank:0.1 nn_rank:0.5 nn_rank:1  marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank:0.1 enet_rank lasso_rank --test_func nn_test_zero > logs/0129_corrinter_zero.log &

python -u mlp_predict.py --dataset CorrelatedInteractionSimulation --rank_func nn_rank:0.005 nn_rank:0.05 nn_rank:0.1 nn_rank:0.5 nn_rank:1  marginal_rank rf_rank zero_rank shuffle_rank random_rank dfs_rank:0.1 enet_rank lasso_rank --test_func nn_test_retrain > logs/0129_corrinter_retrain.log &

python -u mlp_predict.py --dataset YearMSD --identifier 0129 \
--rank_func nn_rank:1  --test_func nn_test_retrain

#
python -u mlp_predict.py --dataset CorrelatedInteractionSimulation --rank_func nn_rank:0.1 rf_rank zero_rank marginal_rank shuffle_rank random_rank dfs_rank:0.1 enet_rank lasso_rank --test_func nn_test_zero --visdom_enabled

python -u mlp_predict.py --dataset NoInteractionSimulation --rank_func dfs_rank:0.1 other_ranks --test_func nn_test_zero --visdom_enabled

python -u mlp_predict.py --dataset CorrelatedNoInteractionSimulation --rank_func nn_rank:1 nn_rank:0.1 nn_rank:0.5 dfs_rank:1 other_ranks --test_func nn_test_zero

# Simulation of linear model
python -u simulate.py --identifier 0111 --dataset GaussSimulation --mode correlation --rank_func vbd_linear_rank:1e-3 dfs_rank:1e-3 other_ranks

python -u mlp_predict.py --dataset ARCENE --identifier 0327 --rank_func nn_joint_rank:1e-3 nn_joint_rank:1e-4 rf_rank lasso_rank enet_rank random_rank --test_func svm_linear_test
python -u mlp_predict.py --dataset GISETTE --identifier 0327 --rank_func nn_joint_rank:1e-2 nn_joint_rank:1e-3 rf_rank lasso_rank enet_rank random_rank --test_func svm_linear_test

python -u mlp_predict.py --dataset MADELON --identifier 0327 --rank_func nn_joint_rank:1e-2 nn_joint_rank:1e-3 rf_rank lasso_rank enet_rank random_rank --test_func svm_linear_test

python -u mlp_predict.py --dataset ARCENE --identifier 0327_low_dropout --rank_func nn_joint_rank:1e-3 nn_joint_rank:1e-4 --test_func svm_linear_test
python -u mlp_predict.py --dataset ARCENE --identifier 0327 --rank_func nn_joint_rank:1e-2 --test_func svm_linear_test

python -u mlp_predict.py --dataset MADELON --identifier 0327 --rank_func nn_joint_rank:1 nn_joint_rank:0.1 --test_func svm_linear_test

python -u mlp_predict.py --dataset DOROTHEA --identifier 0327 --rank_func nn_joint_rank:1e-5 nn_joint_rank:1e-4 nn_joint_rank:1e-3 rf_rank lasso_rank enet_rank random_rank --test_func svm_linear_test

python -u mlp_predict.py --dataset DEXTER --identifier 0327 --rank_func nn_joint_rank:0.1 nn_joint_rank:1e-2 nn_joint_rank:1e-3 rf_rank lasso_rank enet_rank random_rank --test_func svm_linear_test

# Test that if not joint training, what's the result?
python -u mlp_predict.py --dataset ARCENE --identifier 0327 --rank_func nn_rank:1e-1 marginal_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset GISETTE --identifier 0327 --rank_func nn_rank:1e-2 marginal_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset MADELON --identifier 0327 --rank_func nn_rank:1 marginal_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset DEXTER --identifier 0327 --rank_func nn_rank:1e-3 random_rank marginal_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset DOROTHEA --identifier 0327 --rank_func nn_rank:1e-2 nn_rank:1e-1 nn_joint_rank:1e-2 nn_joint_rank:1e-1 marginal_rank mim_rank --test_func svm_linear_test


arr=[]
python -u mlp_predict.py --dataset ARCENE --identifier 0327 --rank_func nn_rank:1e-1 marginal_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset GISETTE --identifier 0327 --rank_func nn_rank:1e-2 marginal_rank mim_rank --test_func svm_linear_test

array=("GISETTE")
dropout_arr=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
for dropout in ${dropout_arr[@]}
do
  python -u mlp_predict.py --dataset GISETTE --identifier 0408 --rank_func "nn_joint_rank:1e-2|$dropout" --test_func svm_linear_test
done

python -u mlp_predict.py --dataset ARCENE --identifier 0327 --rank_func marginal_rank random_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset GISETTE --identifier 0327 --rank_func marginal_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset MADELON --identifier 0327 --rank_func marginal_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset DOROTHEA --identifier 0327 --rank_func marginal_rank mim_rank --test_func svm_linear_test && python -u mlp_predict.py --dataset DEXTER --identifier 0327 --rank_func marginal_rank mim_rank --test_func svm_linear_test

# 0503 back! Almost forget everything lol!!!


python -u mlp_predict.py --dataset ARCENE --identifier 0327 --rank_func mim_rank marginal_rank --test_func svm_linear_test &&
python -u mlp_predict.py --dataset GISETTE --identifier 0327 --rank_func mim_rank marginal_rank --test_func svm_linear_test &&
python -u mlp_predict.py --dataset MADELON --identifier 0327 --rank_func mim_rank marginal_rank --test_func svm_linear_test &&
python -u mlp_predict.py --dataset DOROTHEA --identifier 0327 --rank_func mim_rank marginal_rank --test_func svm_linear_test &&
python -u mlp_predict.py --dataset DEXTER --identifier 0327 --rank_func mim_rank marginal_rank --test_func svm_linear_test

python -u mlp_predict.py --dataset NoInteractionSimulation --rank_func enet_rank lasso_rank --test_func nn_test_zero --no_rank_cache --no_nn_cache > logs/0507_nointer_zero.log &&
python -u mlp_predict.py --dataset InteractionSimulation --rank_func enet_rank lasso_rank --test_func nn_test_zero --no_rank_cache --no_nn_cache > logs/0507_inter_zero.log

## Run iter zero!? For RNN even or other methods. But does not seem to be great!
# If the feature size is small, I should really compare to iterative zero approach!!
python -u mlp_predict.py --dataset support2 --identifier 0111 --rank_func iter_zero_rank --test_func nn_test_zero --no_rank_cache &&
python -u mlp_predict.py --dataset MiniBooNE --identifier 0111 --rank_func iter_zero_rank --test_func nn_test_zero --no_rank_cache &&
python -u mlp_predict.py --dataset OnlineNewsPopularity --identifier 0111 --rank_func iter_zero_rank --test_func nn_test_zero --no_rank_cache &&

./srun.sh -o logs/0508_iter_zero_support2 python -u mlp_predict.py --dataset support2 --identifier 0111 --rank_func iter_shuffle_rank iter_zero_rank --test_func nn_test_zero --no_rank_cache &&
./srun.sh -o logs/0508_iter_zero_MiniBooNE python -u mlp_predict.py --dataset MiniBooNE --identifier 0111 --rank_func iter_shuffle_rank iter_zero_rank --test_func nn_test_zero --no_rank_cache &&
./srun.sh -o logs/0508_iter_zero_OnlineNewsPopularity python -u mlp_predict.py --dataset OnlineNewsPopularity --identifier 0111 --rank_func iter_shuffle_rank iter_zero_rank --test_func nn_test_zero --no_rank_cache &&
./srun.sh -o logs/0508_iter_zero_YearMSD python -u mlp_predict.py --dataset YearMSD --identifier 0111 --rank_func iter_shuffle_rank iter_zero_rank --test_func nn_test_zero --no_rank_cache &&

# Run everything with Lasso and Enet again!!
python -u mlp_predict.py --dataset support2 --rank_func enet_rank lasso_rank --test_func nn_test_zero --no_rank_cache > logs/0509_rerunenetlas_support2_zero.log &
python -u mlp_predict.py --dataset MiniBooNE --rank_func enet_rank lasso_rank --test_func nn_test_zero --no_rank_cache > logs/0509_rerunenetlas_miniboone_zero.log &
python -u mlp_predict.py --dataset OnlineNewsPopularity --rank_func enet_rank lasso_rank --test_func nn_test_zero --no_rank_cache > logs/0509_rerunenetlas_onlinenews_zero.log &
python -u mlp_predict.py --dataset YearMSD --rank_func enet_rank lasso_rank --test_func nn_test_zero --no_rank_cache > logs/0509_rerunenetlas_yearmsd_zero.log &

CUDA_VISIBLE_DEVICES=3 python -u mlp_predict.py --dataset MIMIC_new --rank_func nn_rank:0.1 nn_rank:0.01 nn_rank:1e-3 zero_rank shuffle_rank iter_zero_rank iter_shuffle_rank --test_func nn_test_zero --no_rank_cache &> logs/0513_mimic_new.log &

python -u mlp_predict.py --dataset MIMIC_new --rank_func rf_rank random_rank enet_rank lasso_rank --test_func nn_test_zero --no_rank_cache &> logs/0513_mimic_others.log &

CUDA_VISIBLE_DEVICES=3 python -u mlp_predict.py --dataset MIMIC_new --rank_func nn_rank:1 nn_rank:0.5 nn_rank:0.05  nn_rank:5e-3 nn_rank:0.1 nn_rank:0.01 nn_rank:1e-3 zero_rank shuffle_rank iter_zero_rank iter_shuffle_rank --test_func nn_test_zero --no_rank_cache &> logs/0513_mimic_new.log &

# Run on CA with 2e-4!!


# Run more on BBMP random and blurry on 1e-3 situation!