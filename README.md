# Gradient Rollback Source Code

This repository contains code for the paper [Explaining Neural Matrix Factorization with Gradient Rollback](https://arxiv.org/abs/2010.05516) (Lawrence, Sztyler & Niepert, 2020).

Gradient Rollback experiments consist of 3 steps:

1. Train a matrix factorization model using DistMult or ComplEx & writing influence maps for GR
2. Given a trained model and to-be-explained triples, use GR to identify all relevant explanations &
   extract the top-k if desired.
3. Evaluate either GR or the baseline (NH) 

The 3 steps are explained in more detail below, followed by instruction on how to reproduce the 
results of the paper.

If you have any questions, feel free to reach out!

## Datasets
The GR paper runs experiments on 3 datasets (Nations, FB15k-237, Movielens):
* [Nations](https://github.com/ZhenfengLei/KGDatasets/tree/master/Nations)
* [FB15-237](https://github.com/ZhenfengLei/KGDatasets/tree/master/FB15k-237)
* [Movielens](https://grouplens.org/datasets/MovieLens/100k/)
  
## Dependencies
All dependencies are install if ``python setup.py install`` or ``python setup.py develop`` is run. Please ensure that `pip` and `setuptools` are uptodate by running `pip install --upgrade pip setuptools`.

Additionally, we modified the following tensorflow 2 files:
* REQUIRED! [Embeddings](https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/layers/embeddings.py#L134): In TF 2.2, the Embedding layer is run on the CPU, which leads to drastically slower train times. To modify, remove the if statement in the build function so that always else is executed. Place the resulting file with name ``custom_embeddings`` under ``gr/model``.
* [Adam](https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/optimizer_v2/adam.py#L32-L274): This file can be modified so that the exact gradients are recorded for GR. The current implementation simply recovers the gradients by comparing the weights before and after the update. This could however lead to some rounding errors. 

## Training
To run training and GR code, see steps 1 and 2 below.

## Evaluation
* To run evaluation, see step 3 below.
* Note that results can slightly differ from the ones in the paper due to 
  [known GPU computation imprecision](https://stackoverflow.com/questions/58441514/why-is-tensorflow-2-much-slower-than-tensorflow-1).
  
## Evaluation Metrics
* Probability Drop (PD): Whether the removal of the explanations for a prediction lead to a lower probability for the prediction.
* Top-1 Change (TC): Whether the removal of the explanations for a previously top-1 prediction lead to a different top-1 prediction. This measures the ability of a method to perform removal attacks. For GR, the more stable a model is (e.g. using unitnorm) the more accuract GR is. But at the same time it makes it more difficult to "attack" the model with a removal (the TC values). For example on nations unitnorm versus no constraint (smaller empirical Lipschitz constant indicates more stable model):

    | Method        | Lipschitz           | TC  |
    | ------------- |:-------------| :-----|
    | Unitnorm | 0.67 | 18 |
    | No constraint | 6.93 | 38 |

## Individual Steps
### Step 1
This step trains a main model and tracks the influence of the training set, the corresponding output
file will be used by GR in step 2. The main entry point for this step is the file 
``run_step_1_train_main_model.py``. Example bash files to start this step can be found in 
``bash/*/*step_1_train_main.sh``, where * are placeholders for the dataset name. 
The output is stored in a new generated folder that has the same name as the selected dataset. 
To change from DistMult to ComplEx, edit ``bash/*/*step_1_train_main.sh`` by replacing 
``DistMult`` with ``ComplEx``.

### Step 2
This step allows GR to explain prediction, given a model from step 1, the influence file and the 
triples one wants to explain. The step consists of two sub-steps. The first extracts for each triple
one wants to explain all training instances that are explanations and the corresponding influence score.
In the second, the file from the first sub-step can be further refined by extracting the explanations
with the top-``k`` highest influence score, where ``k`` is a parameter (10 and 100 by default).
The main entry point for the first sub-step is the file ``run_step_2_get_explanations.py``. 
The main entry point for the second sub-step is the file ``run_step_2_extract_topk_explanations.py``.
Example bash files to start both sub-steps can be found in ``bash/*/*step_2_run_GR.sh``, 
where * are placeholders for the dataset name. The output is stored in the same folder as in Step 1.

### Step 3
This step evaluates either GR or a baseline (NH). The step can be quite expensive, because a new
model has to be trained for each triple and its explanation that one wants to evaluate.
The main entry point for this step is the file ``run_step_3_xai_evaluation.py``. 
Example bash files to start this step can be found in  ``bash/*/*step_3_eval_GR.sh`` and 
``bash/*/*step_3_eval_NH.sh``, for GR/NH, respectively, and where * are placeholders for the dataset name.

## Reproduce Results: Example for Nations
* Step 1: ``./bash/nations/nations_step_1_train_main.sh`` or alternatively use the pretrained model
  contained in the folder ``pretrained_models/nations``
* Step 2: ``./bash/nations/nations_step_2_run_GR.sh``
* Step 3: 
  * NH: Run ``./bash/nations/nations_step_3_eval_NH.sh`` with the modification mentioned below for 
  the different setups.
    * NH-1: make sure the bash script has ``K=1`` uncommented
    * NH-10: make sure the bash script has ``K=10`` uncommented
    * NH-all: make sure the bash script has ``K=-1`` uncommented
  * GR: Run ``./bash/nations/nations_step_3_eval_GR.sh`` with the modification mentioned below for 
  the different setups.
    * GR-1: make sure the bash script has ``SUFFIX='.topk.1'`` uncommented
    * GR-10: make sure the bash script has ``SUFFIX='.topk.10'`` uncommented
    * GR-all: make sure the bash script has ``SUFFIX=''`` uncommented
  * Each run will write a log file with the names as indicated in the bash scripts 
    (in general they start with ``info.eval`)
  * To get the values reported in the table (* indicates the suffix of the log file of interest):
    * ``grep 'Score evaluation_all_PD' info.eval*``
    * ``grep 'Score evaluation_all_TC' info.eval*``
  * Results on Nations (identical to Nations' results in Table 1 of the paper)
  
    | Method        | PD           | TC  |
    | ------------- |:-------------| :-----|
    | NH-1 | 54 | 18 |
    | NH-10 | 66 | 36 |
    | NH-all | 82 | 70 |
    | GR-1 | 93 | 38 |
    | GR-10 | 97 | 83 |
    | GR-all | 100 | 97 |
