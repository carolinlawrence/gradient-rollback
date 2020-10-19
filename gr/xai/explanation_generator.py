# coding: utf-8
#        Gradient Rollback
#
#   File:     explanation_generator.py
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
Given a matrix factorization models and its influence map (saved during training, step 1),
this script returns explanations for a set of provided triples that are to be explained.
"""

# standard libraries
import logging
import os
from collections import defaultdict

import numpy as np

# local libraries
from gr.model.model_handler import ModelHandler
from argparse import Namespace
from gr.utils.read_write import read_lines_in_list, read_json, write_json_to_file, \
    write_list_to_file, get_predictions_score_file
from gr.xai.model_trainer import load_dataset, get_predictions, run_prediction
# init GPU parameter, seed, etc.
from gr.utils import utils as utils

LOGGER = logging.getLogger(__name__)


def get_explanation(prog_args: Namespace):
    """
    Given a matrix factorization models and its influence map (saved during training, step 1),
    this script returns explanations for a set of provided triples that are to be explained.
    :param prog_args: instantiate by calling program_arguments, make_xai_parser(2)
    :return: 0 on success, explanation file is written to disc at location prog_args.output_dir
    """
    # Load data
    LOGGER.info("Loading data")
    # Load the file of triples that we want to explain (e.g. top-1/most likely predictions of the main model)
    triples_to_explain, prediction_prob = get_predictions_score_file(prog_args.triples)
    # Load original training and validation data: we need this to be able to built the original
    # entity and relation matrices
    train_data, _, valid_data_original = load_dataset(prog_args.dataset)
    valid_data, _, _, test_skip_index = \
        utils.align_data(train_data, valid_data_original, triples_to_explain)
    correct_ids = None
    if prog_args.correct_ids is not None:
        correct_ids = read_lines_in_list(prog_args.correct_ids)
        correct_ids = np.array(correct_ids, dtype=np.int)
    # Set up the model exactly the same way as the original model was, then load the weights of the original model
    LOGGER.info("Initializing model")
    keep_k_first = None
    if prog_args.dataset == 'movielens':
        keep_k_first = 5
    precision_type = np.float64
    if prog_args.switch_float32 is True:
        precision_type = np.float32
    LOGGER.info("Precision: %s" % precision_type)
    utils.initialize(gpu=prog_args.gpu, deterministic=prog_args.deterministic, switch_float32=prog_args.switch_float32)
    model_holder_params = read_json(prog_args.params)
    model_holder = ModelHandler(experts=model_holder_params['experts'],
                  triples_train=train_data,
                  triples_validation=valid_data,
                  batch_size=int(model_holder_params['batch_size']),
                  epochs=int(model_holder_params['epochs']),
                  num_negative=int(model_holder_params['num_negative']),
                  validation_step=int(model_holder_params['validation_step']),
                  output_dir=prog_args.output_dir,
                  compute_mode='cpu' if prog_args.deterministic else 'gpu',
                  embedding_dim=int(model_holder_params['embedding_dim']),
                  keep_k_first=keep_k_first)
    success = model_holder.load_model(os.path.join(prog_args.output_dir, 'model.model'))
    assert success
    # Load the influence map that we saved during training of the main model,
    # where we saved for each training instance how much it influence each entity/relation
    # In the saved map we have keys of the shape: head:relation:tail:e/r:entity/relation
    # where head:relation:tail is a training triple
    # and e/r indicates that what follows is an entity/relation, respectively
    # entity/relation: the actual entity/relation
    # the value is how much the training triple influence the entity/relation
    influence_map = read_json(prog_args.influence_map)
    lookup_relevant_training_triples = defaultdict(list)
    # we re-write the original influence map for easier access
    modified_influence_map = defaultdict(dict)
    for key in list(influence_map):
        elements = key.split('\t')
        # The first three parts are the training triple
        # The last element is the entity/relation
        # The second last lets us know if its entity or relation
        entity_dim = '%s:%s' % (elements[-2], elements[-1])
        triple = '%s:%s:%s' % (elements[0], elements[1], elements[2])
        # here we re-write the shape of the map
        # we want to look up for an explanation what its influence is the different entities/relations
        # this could be any dimension, but in praxis it should just be its own head/relation/tail
        # outer key is the explanation triple
        # inner key is the entity/relation it effects
        # value is the actual influence score
        modified_influence_map[triple][entity_dim] = np.array(influence_map[key], dtype=precision_type)
        # here we create another dictionary so that for the prediction to explain, we look up
        # all relevant training instances, which we will then pass as the keys we are interested in
        # to get the influence from the modified influence map.
        lookup_relevant_training_triples[entity_dim].append(triple)
        influence_map.pop(key, None)
    influence_map = modified_influence_map

    # keys will be triple name of predictions, values will be a json structure with explanations for this triple
    explanation_dictionary = defaultdict(dict)

    for counter, triple in enumerate(triples_to_explain):
        # if the tail is an empty string, then the main model didn't make a prediction for that triple,
        # this is because either head or relation where unknown to the model.
        # To keep the triple in the output file, we here write an empty list for this triple.
        if triple[2] == '':
            triple_name = " ".join([triple[0], triple[1]])
            explanation_dictionary[triple_name]['explanation'] = []
            continue

        nodes_idx = np.array([model_holder._gp.vertex_indexer[triple[0]]])
        relations_idx = np.array([model_holder._gp.relation_indexer[triple[1]]])
        gold_idx = np.array([model_holder._gp.vertex_indexer[triple[2]]])

        explanation_for_current_triple = defaultdict(float)
        top1_for_current_triple_step_1 = defaultdict(int)
        top1_for_current_triple_step_2 = defaultdict(int)
        # Get all training triples that might have influenced the current prediction triple
        relevant_to_head = lookup_relevant_training_triples['e:' + triple[0]]
        relevant_to_tail = lookup_relevant_training_triples['e:' + triple[2]]
        relevant_to_rel = lookup_relevant_training_triples['r:' + triple[1]]
        all_relevant_training_triples = list(set().union(relevant_to_head,
                                                         relevant_to_tail,
                                                         relevant_to_rel))
        #all_relevant_training_triples.sort()  #this can be switched on to guarantee the same order for the output everytime, but can slow things down
        LOGGER.info("Number of relevant triples: %s" % len(all_relevant_training_triples))
        # For each training triple that might have influence our prediction triple,
        # remove its influence and recompute the score of the prediction triple,
        # get difference to original score
        for training_triple in all_relevant_training_triples:
            influence_of_triple = influence_map[training_triple]
            training_triple_elements = training_triple.split(':')
            influence_head = influence_of_triple['e:'+training_triple_elements[0]]
            influence_rel = influence_of_triple['r:'+training_triple_elements[1]]
            influence_tail = influence_of_triple['e:'+training_triple_elements[2]]
            training_head_idx = model_holder._gp.vertex_indexer[training_triple_elements[0]]
            training_relations_idx = model_holder._gp.relation_indexer[training_triple_elements[1]]
            training_tail_idx = model_holder._gp.vertex_indexer[training_triple_elements[2]]
            scores_original = model_holder._model.predict_tail((nodes_idx, relations_idx)).numpy()
            if keep_k_first is not None:
                scores_original = scores_original[:, :keep_k_first]
            probs_original = utils.softmax(scores_original, axis=1)
            # Remove influence of current training triple
            model_holder._model.update_according_to_influence(
                (training_head_idx, training_relations_idx, training_tail_idx),
                (influence_head, influence_rel, influence_tail))
            # Get scores where the weights are modified to not have the influence of current training triple & turn to probs if needed
            scores_minus_influence = model_holder._model.predict_tail((nodes_idx, relations_idx)).numpy()
            if keep_k_first is not None:
                scores_minus_influence = scores_minus_influence[:, :keep_k_first]
            probs_minus_influence = utils.softmax(scores_minus_influence, axis=1)
            difference = probs_original[0][gold_idx] - probs_minus_influence[0][gold_idx]
            # if removing the influence causes a higher probability,
            # then this will be negative and this was clearly not an explanation triple
            # so we only want to save anything that caused a drop in probability
            if difference >= 0:
                # the closer to zero the difference is, the better
                # e.g. original probability is 1, new is 0, i.e. the training triple fully
                # explains the prediction, then the difference is 1
                explanation_for_current_triple[training_triple] = difference
                top1_for_current_triple_step_1[training_triple] = np.argmax(probs_original[0])+1
                top1_for_current_triple_step_2[training_triple] = np.argmax(probs_minus_influence[0])+1
            # revert to original weights, ready for the next for loop step
            model_holder._model.revert_weights()
        # save triple+prediction of main model+prediction with softmax change and its explanation in a dictionary
        triple_name = " ".join([triple[0], triple[1], triple[2]])
        example_correct = ''
        if correct_ids is not None:
            example_correct = '0'
            if counter in correct_ids:
                example_correct = '1'
        # test sets can contain the same (head, rel) and if we want to explain the argmax of the main model,
        # then its always the same tail and this tail could be correct for one instance in the test set,
        # but wrong for the other. here we ensure that if its is correct at least once, that we record this as correct
        if 'correct' in explanation_dictionary[triple_name]:
            if explanation_dictionary[triple_name]['correct'] == '1':
                continue
        explanation_dictionary[triple_name]['correct'] = example_correct
        explanation_dictionary[triple_name]['number_influences'] = str(len(all_relevant_training_triples))
        explanation_dictionary[triple_name]['probability'] = str(round(probs_original[0][gold_idx][0], 5))
        if len(explanation_for_current_triple) > 1:
            LOGGER.info("Saving explanations for this triple: %s" % triple)
            LOGGER.info("Number of possible explanations: %s" % len(explanation_for_current_triple))
            all_explanations = [None] * len(explanation_for_current_triple)
            for i, (key, value) in enumerate(explanation_for_current_triple.items()):
                explanation_triple = key.split(":")
                all_explanations[i] = (explanation_triple, float(value), 'a', )  # 'a' is an artifact from a previous solution, serves no purpose anymore
            explanation_dictionary[triple_name]['explanation'] = all_explanations
        else:
            LOGGER.warning("Check this triple: %s" % triple)
            explanation_dictionary[triple_name]['explanation'] = []
        explanation_dictionary[triple_name]['number_explanations'] = str(len(explanation_dictionary[triple_name]['explanation']))

    LOGGER.info("Writing explanations to disc")
    # write the explanation dictionary to disk
    write_json_to_file(explanation_dictionary,
                       prog_args.triples+'.explanations')
    return 0