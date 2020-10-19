# coding: utf-8
#        Gradient Rollback
#
#   File:     model_trainer.py
#   Authors:  Carolin Lawrence carolin.lawrence@neclab.eu
#             Timo Sztyler timo.sztyler@neclab.eu
#             Mathias Niepert mathias.niepert@neclab.eu
#
# NEC Laboratories Europe GmbH, Copyright (c) 2020, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#
"""
Trains a main model or models with certain training instances missing for evaluation (step 1 & 3)
"""

# standard libraries
import logging
import os
import time
import gc
import random
from collections import defaultdict
from argparse import Namespace

import numpy as np
from tensorflow.keras import backend as K

# local libraries
from gr.model.model_handler import ModelHandler
from gr.utils.read_write import write_list_to_file, \
    read_lines_in_list, read_json, get_predictions_score_file, read_lines_in_list_as_ints
from gr.xai.delete_facts import FactDeleter, get_fact_deleter, ExplanationDeleter

# init GPU parameter, seed, etc.
from gr.utils import utils

LOGGER = logging.getLogger(__name__)


def get_predictions(predict_data: np.array, scores: np.array, ranks: np.array, entities: list,
                    predict_skip_index: list, k: int = 1, gold_tails: bool = False):
    """
    Given prediction data and scores and entities returned from ModelHandler's predict method,
    create the predictions that are written for xai.

    :param predict_data: (n, 3)-array, where n is the number of predict instances, each
            containing head, relation, gold tail
    :param scores: (n, m)-array, where m is the number of possible entities
            the score can be either a score or a probability distribution
            (if the scores where softmaxed)
    :param ranks: the rank of the gold tail
    :param entities: list of size m, each entry is a possible entity
            and the index corresponds to the m column of scores
    :param predict_skip_index: ModelHandler cannot predict if part of the tuple is not in the training data.
        This list is the size of the original predict set, before these problematic tuples were removed
        It contains a 1 if that index was deleted, else 0.
    :param k: write the top-k predictions
    :param gold_tails: if true, look up the score of the gold prediction
    :return: list of size n, each entry is a tuple consisting of a list,
            each entry of this list is a prediction (for k=1 and gold_tails=True,
            the list will only have 1 entry) for the n-th example in the form of a tuple:
            (head, relation, kth prediction / gold tail if gold_tails=True, score)

    """
    # sort and build the result
    result = []
    one_if_top1_correct = []
    scores_idx = 0
    # loop over original set of nodes to keep track of the original set while adding the scores
    # of the instances where both node and relation were found
    entities_dict = {entity: i for i, entity in enumerate(entities)}
    LOGGER.info('Size predict skip index: %s' % len(predict_skip_index))
    counter = 0
    for idx, entry in enumerate(predict_skip_index):
        head, rel, gold_tail = predict_data[idx]
        if entry == 0:  # then prediction was made
            # Hint: score is a 1d numpy array of size |nodes|
            score = scores[scores_idx]
            rank_of_gold = ranks[scores_idx]
            top_score_idx = int(np.argmax(np.array(score)))
            if entities[top_score_idx] == gold_tail:
                counter += 1
                one_if_top1_correct.append(1)
            elif rank_of_gold == 1:
                LOGGER.warning((head, rel, gold_tail))
                LOGGER.warning(entities[top_score_idx])
                LOGGER.warning(entities)
                LOGGER.warning(score)
                LOGGER.warning(top_score_idx)
                raise ValueError('This shouldn\'t happen. Please Debug. Is your gold tail maybe '
                                 'a known training triple? Because known training triples receive'
                                 'a score of -math.inf in model_handler\'s predict')
            else:
                one_if_top1_correct.append(0)
            score = np.around(score, decimals=5)
            if gold_tails is True:
                score_gold_tail = score[entities_dict[gold_tail]]
                # even though there is only one entry in the list, we still use it to stay compatible
                # with __predict_nodes which returns the top-k in the list
                result.append([(head, rel, gold_tail, score_gold_tail)])
            elif k == 1:
                result.append([(head, rel, entities[top_score_idx], score[top_score_idx])])
            else:
                score = np.array(score)
                top_score_idxs = np.argpartition(score, -k)[-k:]
                sorted_top_score_idxs = top_score_idxs[np.argsort(score[top_score_idxs])][::-1]
                result.append(
                    [(head, rel, entities[current_idx], score[current_idx])
                     for current_idx in sorted_top_score_idxs])
            scores_idx += 1
        else:  # no prediction exists, write dummy result
            result.append([(head, rel, '', -1)])
            one_if_top1_correct.append(0)
    return result, one_if_top1_correct


def print_prediction(prog_args, predictions, one_if_top1_correct, suffix):
    """
    Given a set of predictions and whether each prediction is correct or not, write them to
    several files.
    :param prog_args: instance of :py:class:ArgumentParser return from make_xai_parser(step=1/3)
    :param predictions: a list of a list of predictions, outer list: different head/rels, inner list
        is top-k of predictions for this head/rel. here we only write the top1
    :param one_if_top1_correct: indicates whether the corresponding prediction is correct or not
    :param suffix: potentially pass a suffix to append to file names
    :return: 0 on success
    """
    assert len(predictions) == len(one_if_top1_correct)
    top1_all_predictions = []
    top1_correct_predictions = []
    top1_wrong_predictions = []
    top1_all_predictions_scores = []
    top1_correct_predictions_scores = []
    top1_wrong_predictions_scores = []
    top1_correct_ids = []
    for i, topk in enumerate(predictions):
        top1 = topk[0]
        top1_all_predictions_scores.append(top1)
        top1_all_predictions.append(top1[0] + '\t' + top1[1] + '\t' + top1[2])
        if one_if_top1_correct[i] == 1:
            top1_correct_predictions.append(top1[0] + '\t' + top1[1] + '\t' + top1[2])
            top1_correct_predictions_scores.append(top1)
            top1_correct_ids.append(i)
        elif one_if_top1_correct[i] == 0:
            top1_wrong_predictions.append(top1[0] + '\t' + top1[1] + '\t' + top1[2])
            top1_wrong_predictions_scores.append(top1)
    LOGGER.info('Writing predictions to disc.')
    write_list_to_file(top1_all_predictions_scores,
                       prog_args.output_dir + "top1_all_predictions_scores.txt" + suffix)
    write_list_to_file(top1_all_predictions,
                       prog_args.output_dir + "top1_all_predictions.txt" + suffix)
    write_list_to_file(top1_correct_predictions,
                       prog_args.output_dir + "top1_correct_predictions.txt" + suffix)
    write_list_to_file(top1_wrong_predictions,
                       prog_args.output_dir + "top1_wrong_predictions.txt" + suffix)
    write_list_to_file(top1_correct_predictions_scores,
                       prog_args.output_dir + "top1_correct_predictions_scores.txt" + suffix)
    write_list_to_file(top1_wrong_predictions_scores,
                       prog_args.output_dir + "top1_wrong_predictions_scores.txt" + suffix)
    write_list_to_file(top1_correct_ids,
                       prog_args.output_dir + "correct_ids" + suffix)
    return 0


def run_prediction(prog_args: Namespace, model_holder: ModelHandler, predict_data: np.array,
                   predict_skip_index: list, predict_data_original: np.array,
                   k: int = 1, scores_as_probabilities: bool = True,
                   lookup_tails: bool = False,
                   write_predictions: bool = True) -> tuple:
    """
    Given an ModelHandler instance and a set of triplets, send head and relation to the predict method
    to get the top k tails
    :param prog_args: instance of :py:class:ArgumentParser return from make_xai_parser(step=1/3)
    :param model_holder: instance of ModelHandler
    :param predict_data: a set of triplets with (head, relation, tail) but we will predict the tail.
    :param predict_skip_index: model_holder cannot predict if part of the tuple is not in the training data.
        This list is the size of the original predict set, before these problematic tuples were removed
        It contains a 1 if that index was deleted, else 0.
    :param predict_data_original: a set of triplets with (head, relation, tail) but we will predict the tail.
            This instance should contain all triples. The knowledge if we skipped an instance comes
            from predict_skip_index
    :param k: how many tail predictions to return per example
    :param scores_as_probabilities: If true,
            then the scores are turned into probabilities via softmax
    :param lookup_tails: if True, then we look up the tails' probabilities rather than predicting
    :param get_details: prints additional statistics
    :return: tuple of:
            predictions: the top 1 predictions
            predictions_lookup: given some gold tails, what is the probability of them?
    """
    LOGGER.info("Starting prediction")
    # atm scores returns tail prediction on every even index and head prediction on odd index
    scores_softmaxed, entities, ranks = \
        model_holder.predict_set(predict_data, mask_known_triples=True,
                      scores_as_probabilities=scores_as_probabilities)
    mrr = utils.print_model_performance(ranks)

    # handle case where softmax is applied before known are set to -math.inf
    predictions, one_if_top1_correct = \
        get_predictions(predict_data_original, scores_softmaxed, ranks, entities, predict_skip_index, k=k)
    predictions_lookup = None
    if lookup_tails is True:
        # get lookup
        predictions_lookup, _ = get_predictions(predict_data_original, scores_softmaxed, ranks, entities,
                                                predict_skip_index, gold_tails=True)

    LOGGER.info("Completed prediction")
    if write_predictions is True:
        LOGGER.info("Writing prediction")
        print_prediction(prog_args, predictions, one_if_top1_correct, '')

    return predictions, predictions_lookup, mrr


def run_train(prog_args: Namespace,
              train_data: list,
              valid_data: list,
              train_data_subset: list = None,
              no_save: bool = False,
              model_params: bool = False,
              delete_idx: list = None) -> ModelHandler:
    """
    Instantiates a ModelHandler object, trains a model and saves it.
    :param prog_args: instance of :py:class:ArgumentParser return from make_xai_parser(step=1/3)
    :param train_data: the training data to build graph and train on
    :param valid_data: the validation data to validate on
    :param train_data_subset: the training data to train on
    :param no_save: if true, it doesn't saves the resulting model
    :param model_params: If True, the model params are loaded from a file, given via prog_args.params
    :return: the ModelHandler object
    """
    model_holder_params = None
    if model_params is True:
        model_holder_params = read_json(prog_args.params)
        seed = int(model_holder_params['seed'])
    else:
        seed = prog_args.seed
    LOGGER.info("Setting seed")
    if seed == -1:
        seed = random.randrange(2 ** 32 - 1)
    LOGGER.info("Seed: %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    # minus one means we assume the cuda variable has been set on the command line previously
    utils.initialize(gpu=prog_args.gpu, seed=seed,
                     deterministic=prog_args.deterministic,
                     switch_float32=prog_args.switch_float32)

    optimizer = None
    if model_params is True:
        if 'Adam' in model_holder_params['optimizer']:
            optimizer = 'adam'
        elif 'SGD' in model_holder_params['optimizer']:
            optimizer = 'sgd'
        learning_rate = float(model_holder_params['optimizer_lr'])
    else:
        optimizer = prog_args.optimizer
        learning_rate = prog_args.learning_rate
    if optimizer == 'sgd':
        from tensorflow.keras.optimizers import SGD
        optimizer = SGD(learning_rate=learning_rate, name='SGD')
    elif optimizer == 'adam':
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=learning_rate, name='Adam')
    else:
        raise ValueError('Unkown arguments for argument --optimizer.')

    LOGGER.info("Initializing model")
    keep_k_first = None
    if prog_args.dataset == 'movielens':
        keep_k_first = 5
    if model_params is True:
        LOGGER.info('Using model params')
        train_with_softmax = True
        influence_flag = False
        if 'train_with_softmax' in model_holder_params:
            if model_holder_params['train_with_softmax'] == 'True':
                train_with_softmax = True
                LOGGER.info("train_with_softmax: %s" % train_with_softmax)
            else:
                LOGGER.info("train_with_softmax: %s" % train_with_softmax)
        model_holder = ModelHandler(experts=model_holder_params['experts'],
                      triples_train=train_data,
                      triples_train_subset=train_data_subset,
                      triples_validation=valid_data,
                      batch_size=int(model_holder_params['batch_size']),
                      epochs=int(model_holder_params['epochs']),
                      num_negative=int(model_holder_params['num_negative']),
                      validation_step=(int(model_holder_params['epochs'])+1),  # we don't want to waste time on validation for evaluation
                      output_dir=prog_args.output_dir,
                      optimizer=optimizer,
                      compute_mode='cpu' if prog_args.deterministic else 'gpu',
                      embedding_dim=int(model_holder_params['embedding_dim']),
                      print_model_summary=False if no_save is True else True,
                      seed=seed,
                      deterministic=prog_args.deterministic,
                      keep_k_first=keep_k_first,
                      switch_float32=prog_args.switch_float32,
                      train_with_softmax=train_with_softmax,
                      delete_idx=delete_idx)
    else:
        if prog_args.skip_in_training != '':
            delete_idx = read_lines_in_list_as_ints(prog_args.skip_in_training)
            LOGGER.info('Skipping training instances as indicated in: %s' % prog_args.skip_in_training)
        influence_flag = True
        model_holder = ModelHandler(experts=[prog_args.expert],
                      triples_train=train_data,
                      triples_train_subset=train_data_subset,
                      triples_validation=valid_data,
                      batch_size=prog_args.batch_size,
                      epochs=prog_args.epochs,
                      num_negative=prog_args.num_negative,
                      validation_step=prog_args.validation_step,
                      output_dir=prog_args.output_dir,
                      optimizer=optimizer,
                      compute_mode='cpu' if prog_args.deterministic else 'gpu',
                      embedding_dim=prog_args.latent_expert_embedding_dim,
                      print_model_summary=False if no_save is True else True,
                      seed=seed,
                      deterministic=prog_args.deterministic,
                      keep_k_first=keep_k_first,
                      switch_float32=prog_args.switch_float32,
                      train_with_softmax=prog_args.train_with_softmax,
                      delete_idx=delete_idx)

    save_model = True
    if no_save is True:
        save_model = False
    if prog_args.no_influence is True:
        influence_flag = False
    prefix = 'model'
    LOGGER.info("Starting training")
    LOGGER.info("Influence flag: %s" % influence_flag)
    model_holder.train(save_model=save_model, prefix=prefix, compute_influence_map=influence_flag)
    return model_holder


def train_models_for_each_prediction(data_set, prog_args):
    """
    Function for step 3. This will train a new model for each prediction and its explanation.
    :param data_set: tuple of the original dataset
    :param prog_args: instance of :py:class:ArgumentParser return from make_xai_parser(step=3)
    :return: 0 on success (evaluation score are written to log file)
    """
    triples_from_explanation_file = read_json(prog_args.explanations_file_path)

    train_data, valid_data_original, test_data_original = data_set
    # the following line can delete instances, which means the index in comparison to the original
    # file can be shifted, this is recorded in the *index variables which has the original length
    # and contains 1 if that index has been deleted, else 0
    valid_data, test_data, valid_skip_index, test_skip_index = \
        utils.align_data(train_data, valid_data_original, test_data_original)

    deleter = get_fact_deleter(prog_args, len(train_data), len(triples_from_explanation_file))
    # overall scores
    scores_overall = defaultdict(list)
    num_processed_test_instance = 0

    LOGGER.info("NumTestInstances: %s" % len(triples_from_explanation_file))
    collect_num_deleted_triples = []
    probability_gr_and_true = []

    all_prob_step_1 = []
    all_prob_step_3 = []
    for i, test_instance in enumerate(triples_from_explanation_file):
        info_current_test_instance = triples_from_explanation_file[test_instance]
        start = time.time()
        LOGGER.info("Test instance: %s" % test_instance)
        test_instance = tuple(test_instance.split(" "))
        test_head, test_rel, test_tail = test_instance

        # Remove some triples from the new training data specific to the current triple
        deleter.set_triple_of_interest(test_instance)
        num_processed_test_instance += 1
        delete_idx = deleter.remove_triples(train_data, int(info_current_test_instance['number_explanations']))
        LOGGER.info('Number of delete triples: ' + str(len(delete_idx)))
        collect_num_deleted_triples.append(len(delete_idx))

        if len(delete_idx) == 0:
            LOGGER.info('> Skip! No training instances where deleted.')
            num_processed_test_instance -= 1
            break

        model_holder = run_train(prog_args, train_data, valid_data, no_save=True, model_params=True, delete_idx=delete_idx)
        # by definition, we are only iterating over examples that aren't skipped
        local_test_skip_index = [0]
        triple = np.array([[test_head, test_rel, test_tail]])
        predictions, predictions_lookup, mrr = \
            run_prediction(prog_args, model_holder, triple, local_test_skip_index,
                           triple, scores_as_probabilities=True,
                           lookup_tails=True, write_predictions=False)

        # Evaluation
        def get_evaluation_statistics(from_main_model_prob, from_aux_model, from_aux_model_lookup,
                                      prefix='', correct=None):
            correct_marker = ''
            if correct is True:
                correct_marker = 'c_'
            elif correct is False:
                correct_marker = 'w_'
            LOGGER.info(prefix+correct_marker+'prob step 1:'+str(from_main_model_prob))
            LOGGER.info(prefix+correct_marker+'prob step 3:'+str(from_aux_model_lookup[0][0][3]))
            LOGGER.info(prefix+correct_marker+'top-1 step 1:'+str(from_aux_model_lookup[0][0][:3]))
            LOGGER.info(prefix+correct_marker+'top-1 step 3:'+str(from_aux_model[0][0][:3]))
            #PD: Probability Drop
            pd_counter = 1 if from_aux_model_lookup[0][0][3] < from_main_model_prob else 0
            scores_overall[prefix+correct_marker+'pd_all'].append(pd_counter)
            LOGGER.info(prefix+correct_marker+'pd_all:'+str(pd_counter))
            #TC: Top-1 Change
            tc_counter = 1 if from_aux_model[0][0][:3] != from_aux_model_lookup[0][0][:3] else 0
            scores_overall[prefix+correct_marker+'tc_all'].append(tc_counter)
            LOGGER.info(prefix+correct_marker+'tc_all:'+str(tc_counter))

        example_correct = None
        if info_current_test_instance['correct']:
            if info_current_test_instance['correct'] == '1':
                LOGGER.info('correct instance')
                example_correct = True
            else:
                example_correct = False
                LOGGER.info('wrong instance')
        get_evaluation_statistics(float(info_current_test_instance['probability']), predictions, predictions_lookup, 'evaluation_', example_correct)
        probability_true = predictions_lookup[0][0][3]
        probability_main = float(info_current_test_instance['probability'])
        probability_gr_all = []
        for entry in info_current_test_instance['explanation']:
            probability_gr_all.append((probability_main - float(entry[1])))
        probability_gr_and_true.append((" ".join(test_instance), probability_gr_all, probability_true, probability_main))

        all_prob_step_1.append(probability_main)
        all_prob_step_3.append(predictions_lookup[0][0][3])

        K.clear_session()
        del model_holder
        del delete_idx
        del predictions
        del predictions_lookup
        gc.collect()

        elapsed_time = time.time() - start
        LOGGER.info("Time taken for 1 example: %s" % str(elapsed_time))

    LOGGER.info('Number of processed test instances: %s', str(num_processed_test_instance))

    def print_local_statistics(list_of_statistics: list, name: str, prefix: str, mult=100.0):
        list_of_statistics = np.array(list_of_statistics)
        avg = mult * np.average(list_of_statistics)
        var = mult * np.var(list_of_statistics)
        stddev = mult * np.std(list_of_statistics)
        LOGGER.info('Full list %s%s avg: %s', prefix, name, list_of_statistics)
        LOGGER.info('Score %s%s avg: %.0f', prefix, name, avg)
        LOGGER.info('%s%s var: %.0f', prefix, name, var)
        LOGGER.info('%s%s stddev: %.0f', prefix, name, stddev)
        return avg

    def print_evaluation_statistics(prefix='', second_prefix=''):
        if second_prefix is not '':
            # if we have two prefixes fixed, we assume that they end in c_ or w_
            prefix_overall = prefix[:-2] + 'all_'
            scores_overall[prefix_overall + 'pd_all'] = scores_overall[prefix+'pd_all'] + scores_overall[second_prefix+'pd_all']
            scores_overall[prefix_overall + 'tc_all'] = scores_overall[prefix+'tc_all'] + scores_overall[second_prefix+'tc_all']
            prefix = prefix_overall
        print_local_statistics(scores_overall[prefix+'pd_all'], 'PD', prefix)
        print_local_statistics(scores_overall[prefix+'tc_all'], 'TC', prefix)

    #we will get an error if a file contains only correct or only wrong
    print_evaluation_statistics('evaluation_c_')
    print_evaluation_statistics('evaluation_w_')
    print_evaluation_statistics('evaluation_c_', 'evaluation_w_')

    print_local_statistics(collect_num_deleted_triples, 'Deleted', 'Overall ', mult=1.0)
    deleter.print_statistics()
    # Note that this file only makes sense for GR
    write_list_to_file(probability_gr_and_true, prog_args.explanations_file_path+'.probs.output')
    return 0


def load_dataset(dataset):
    """
    Loads a dataset which is assumed to be in the data folder and has a folder name equal to the
    dataset name. Files inside the folder are assumed to be called 'train.txt', 'valid.txt' and 'test.txt'
    :param dataset: the name of the dataset
    :return: a tuple of train, test and valid data
    """
    file_encoding = 'utf8'
    dtype = 'str'
    delimiter = '\t'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_folder = dir_path + '/../../data/' + str(dataset) + '/'

    # loads the training data
    train_file_path = root_folder + 'train.txt'
    train_data = np.loadtxt(train_file_path, encoding=file_encoding, dtype=dtype,
                            delimiter=delimiter, unpack=False)
    # loads the validation data
    valid_file_path = root_folder + 'valid.txt'
    valid_data = np.loadtxt(valid_file_path, encoding=file_encoding, dtype=dtype,
                            delimiter=delimiter, unpack=False)
    # loads the testing data
    test_file_path = root_folder + 'test.txt'
    test_data = np.loadtxt(test_file_path, encoding=file_encoding, dtype=dtype,
                           delimiter=delimiter, unpack=False)

    return train_data, test_data, valid_data


def train_runner(prog_args: Namespace, step) -> int:
    """
    Given the current arguments, train and evaluate a ModelHandler model
    :param prog_args: instance of :py:class:ArgumentParser return from make_xai_parser(step=1/3)
    :return: 0 on success
    """

    LOGGER.info("Loading data")
    train_data, test_data_original, valid_data_original = load_dataset(prog_args.dataset)
    # the following line can delete instances, which means the index in comparison to the original
    # file can be shifted, this is recorded in the *index variables which has the original length
    # and contains 1 if that index has been deleted, else 0
    valid_data, test_data, valid_skip_index, test_skip_index = \
        utils.align_data(train_data, valid_data_original, test_data_original)

    mrr = -1
    if step == 3:  # then we are training auxiliary models
        train_models_for_each_prediction((train_data, valid_data, test_data), prog_args)
    elif step == 1:
        model_holder = run_train(prog_args, train_data, valid_data, no_save=prog_args.no_save)
        LOGGER.info('Starting prediction')
        _, _, mrr = run_prediction(prog_args, model_holder, test_data, test_skip_index,
                                         test_data_original,
                                         scores_as_probabilities=True,
                                         k=prog_args.top_k, write_predictions=True)
    return mrr
