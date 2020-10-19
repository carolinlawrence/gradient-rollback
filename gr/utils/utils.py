# coding=utf-8
#        Gradient Rollback
#
#   File:     utils.py
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
Provides some utility functions.
"""
import os
import random as rn
from collections import namedtuple
from prettytable import PrettyTable
import numpy as np
import logging
import tensorflow as tf

LOGGER = logging.getLogger(__name__)


def initialize(gpu: str = '0', seed: int = 8675309, deterministic: bool = False,
               switch_float32: bool = False):
    """Perform the necessary initialization to ensure semi-deterministic
    results

    This is based on instructions here:
        https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

    :param gpu: The value to pass to `CUDA_VISIBLE_DEVICES` for claiming gpus. This can
        also be a comma-delimited string to grab more than one gpu. If -1, uses whatever was set via
        export CUDA_VISIBLE_DEVICES
    :param seed: the seed
    :param deterministic: If true, run on CPU
    :param switch_float32: If true, use float32, else float64
    :return: 0 on success
    """
    if gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    if deterministic:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # force to use the CPU

    # set the various random seeds
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)
    rn.seed(seed)

    # so the IDs match nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    tf.random.set_seed(seed)

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(len(gpus), "Physical GPUs,")
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_per_process_memory_fraction(gpus[0], 0.19)
    if switch_float32 is False:
        tf.keras.backend.set_floatx('float64')
    return 0


def align_data(train_data: np.array, valid_data: np.array, test_data: np.array) -> (np.array, np.array):
    """
    Embeddings only exist if the entity/relation occurs in the training.
    This function filters out instances of validation and test set if either entriy or the relation
    are unknown.,

    :param train_data: the train data
    :param valid_data: the validation data
    :param test_data: the test data
    :return: a tuple of:
        1. valid_data: the validation data with unknown instances removed
        2. test_data: the test data with unknown instances removed
        3. valid_skip_index: same length as the original valid_data, 1 if the original instance at
           this index is deleted in valid_data, else 0
        3. test_skip_index: same length as the original test_data, 1 if the original instance at
           this index is deleted in test_data, else 0
    """
    known_nodes = set(train_data[:, 0]).union(train_data[:, 2])
    known_relations = set(train_data[:, 1])

    valid_data_cleaned = []
    valid_skip_index = []
    for entry in valid_data:
        if entry[0] not in known_nodes or entry[2] not in known_nodes or entry[1] not in known_relations:
            valid_skip_index.append(1)
            continue
        valid_data_cleaned.append(entry)
        valid_skip_index.append(0)
    valid_data = np.array(valid_data_cleaned)

    test_data_cleaned = []
    test_skip_index = []
    for entry in test_data:
        if entry[0] not in known_nodes or entry[2] not in known_nodes or entry[1] not in known_relations:
            test_skip_index.append(1)
            continue
        test_data_cleaned.append(entry)
        test_skip_index.append(0)
    test_data = np.array(test_data_cleaned)

    return valid_data, test_data, valid_skip_index, test_skip_index


def print_model_performance(ranks: list) -> np.array:
    """
    Given a list of ranks as returned by the predict_set function in model_trainer,
    print the performance statistics

    :param ranks: third return argument of predict_set function in model_trainer.
    :return: the MRR value
    """

    LOGGER.info('Performance')
    table = PrettyTable()
    table.field_names = ['MR', 'MRR', 'Hits@1 [%]', 'Hits@3 [%]', 'Hits@5 [%]', 'Hits@10 [%]']

    ones = np.ones((len(ranks)))
    ranks = np.array(ranks)
    hits10 = np.mean(ranks < 11)
    hits5 = np.mean(ranks < 6)
    hits3 = np.mean(ranks < 4)
    hits1 = np.mean(ranks == 1)
    mr = np.mean(ranks)
    mrr = np.mean(ones / ranks)

    table.add_row([format(mr, '.4f'), format(mrr, '.4f'), format(hits1, '.4f'), format(hits3, '.4f'),
                   format(hits5, '.4f'), format(hits10, '.4f')])
    LOGGER.info('\n'+str(table)+'\n')

    return mrr  # required by the validation step


def softmax(X, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Copied from: https://nolanbconaway.github.io/blog/2017/softmax-numpy
    (but removed theta parameter)

    :param X: numpy array
    :param axis: the axis along which to perform the softmax operation
    :return: numpy array with softmax operation performed
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p




def is_empty(param: list) -> bool:
    if param is None or len(param) == 0:
        return True

    return False


def at_most_one_none(triples: np.array) -> bool:
    mask1 = np.isin(triples[:, 0], [None])
    mask2 = np.isin(triples[:, 1], [None])
    mask3 = np.isin(triples[:, 2], [None])

    return not((mask1 & mask2).any() or (mask1 & mask3).any() or (mask1 & mask2).any())


def uses_only_subset_of_elements(triples: np.array, triples_subset: np.array) -> bool:
    nodes = set(triples[:, 0]).union(triples[:, 2])
    nodes_subset = set(triples_subset[:, 0]).union(triples_subset[:, 2])
    if len(nodes.intersection(nodes_subset)) != len(nodes_subset):
        return False

    relations = set(triples[:, 1])
    relations_subset = set(triples_subset[:, 1])
    if len(relations.intersection(relations_subset)) != len(relations_subset):
        return False

    return True


def load_txt_file(filename: str) -> list:
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            args = line.split('\t')
            args[-1] = args[-1][:-1]
            lines.append(args)
    return lines


def get_known_triples(data: namedtuple, graph_properties: namedtuple) -> set:
    triples = set()

    vertex_indexer = graph_properties.vertex_indexer
    relation_indexer = graph_properties.relation_indexer

    triples = triples.union(extract_triples(data.triples_train, vertex_indexer, relation_indexer))
    triples = triples.union(extract_triples(data.triples_validation, vertex_indexer, relation_indexer))
    return triples


def sigmoid(x: float) -> float:
    return 1. / (1 + np.exp(-x))


def extract_triples(data: np.array, vertex_indexer: dict, relation_indexer: dict) -> set:
    triples = [(vertex_indexer[triple[0]], relation_indexer[triple[1]], vertex_indexer[triple[2]])
               for triple in data
               if triple[0] in vertex_indexer and triple[1] in relation_indexer and triple[2] in vertex_indexer]

    return set(triples)