# coding: utf-8
#        Gradient Rollback
#
#   File:     model_handler.py
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
import collections
import math
import numpy as np
import os
import tqdm
import logging
import time
import copy

from tensorflow.keras import losses
from tensorflow.keras.optimizers import Optimizer, Adam
import tensorflow as tf

from numpy.random import RandomState

import gr.utils.utils as utils
from gr.utils.read_write import write_json_to_file, write_list_to_file
from gr.model.distmult import DistMult
from gr.model.complex import ComplEx

EXPERTS = {
    'DistMult': DistMult,
    'ComplEx': ComplEx
}

LOGGER = logging.getLogger(__name__)

InputParamTuple = collections.namedtuple('InputParamTuple', 'num_negative batch_size epochs validation_step'
                                                            ' loss optimizer optimzer_lr activation_fn expert seed train_with_softmax')
Data = collections.namedtuple('Data',
                              'triples_train triples_train_idx triples_train_subset triples_train_subset_idx triples_validation')
Graph = collections.namedtuple('Graph', 'vertices vertex_indexer num_vertices index_vertex relations'
                                        ' relation_indexer index_relation num_relations')


class ModelHandler(tf.keras.Model):
    """Handles training and prediction of an underlying matrix factorization model.

    Parameters
    ----------
    batch_size : int
        Number of samples per gradient update, default to 256.

    epochs : int
        Number of epochs to train the model, default to 100.

    activation_fn: object
        Keras activation function, default is 'softmax'

    num_negative : int
        Number of negative samples.

    optimizer: optimizers
        optimizer for the keras model, default is 'keras.optimizers.Adam()'

    loss: losses
        Keras loss function, default is 'categorical_crossentropy'

    validation_step: int
        Intervals in which the model is validated, default to 5

    normalize_score: bool
        Whether to normalized the ranking score with a sigmoid function, default is False

    compute_mode: {'cpu', 'gpu'}
        Allows to control whether multiprocessing is activated or deactivate. This parameter shouldn't be specified
        unless the user experience problems, default is 'gpu'

    """
    def __init__(self, experts: list, triples_train: np.array, triples_validation: np.array,
                 triples_train_subset: np.array = None,
                 batch_size: int = 256, epochs: int = 100, activation_fn: str = 'softmax',
                 num_negative: int = 100, optimizer: Optimizer = Adam(),
                 loss: losses = 'categorical_crossentropy', validation_step: int = 5, normalize_score: bool = False,
                 compute_mode: str = 'gpu', output_dir: str = '', embedding_dim: int = 100,
                 print_model_summary: bool = False, seed: int = None,
                 deterministic: bool = False, keep_k_first: int = None, switch_float32: bool = False,
                 train_with_softmax: bool = False, delete_idx: list = None):
        """
        :param experts: A list of which experts to use, e.g. ['DistMult']
        :param triples_train: List of triples to build and train the model. It is assumed that the passed np.array has the shape (n,3).
        :param triples_validation: List of triples to validate the model. All entities and relations have to occur in `triples_train`.
                                    It is assumed that the passed np.array has the shape (n,3).
        :param triples_train_subset: List of triples, if supplied this will be used to train the model, whereaas triples_train will
                                    be used to create the graph.
        :param batch_size:
        :param epochs:
        :param activation_fn:
        :param num_negative:
        :param optimizer:
        :param loss:
        :param validation_step:
        :param normalize_score:
        :param compute_mode:
        :param output_dir:
        :param embedding_dim: Embedding dimension size for DistMult
        :param print_model_summary: If True, print the model summary, else don't
        :param seed: pass the seed to write it to model.params, not actually used any further here
        """

        super().__init__()
        if switch_float32 is True:
            self.tf_precision = tf.float32
            self.np_precision = np.float32
        else:
            self.tf_precision = tf.float64
            self.np_precision = np.float64
        if delete_idx is not None:
            delete_idx.sort()
        self.delete_idx = delete_idx
        self.train_with_softmax = train_with_softmax
        self._deterministic = deterministic
        self._print_model_summary = print_model_summary
        self._keep_k_first = keep_k_first
        self._expert_list = []
        self._data = None
        self._model = None
        # if at the end of training we haven't saved the best model according to validation,
        # make sure the last model is saved
        self._best_model_save = False
        self._normalize_score = normalize_score
        self._networkx_graph = None
        self._output_dir = output_dir
        self._current_epoch = 0
        self._output_experts = []
        self.optimizer = optimizer
        self.global_step = 0
        self.decay_lr = tf.optimizers.schedules.ExponentialDecay(3e-3, 1000, 0.96)

        # the following populates self._gp and self_data
        self.__load_graph(triples_train, triples_validation, triples_train_subset)

        self._model_params = InputParamTuple(num_negative=num_negative,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_step=validation_step,
                                             optimizer=optimizer,
                                             optimzer_lr=str(optimizer.lr.numpy()),
                                             loss=loss,
                                             activation_fn=activation_fn,
                                             expert=experts,
                                             seed=seed,
                                             train_with_softmax=train_with_softmax)

        compute_mode = 'gpu' if compute_mode.lower() != 'cpu' else 'cpu'
        self.__compute_mode = compute_mode.lower()

        # experts is a list but we simply take the first
        self.experts = experts[0]
        self.embedding_dim = embedding_dim
        self._expert_model = EXPERTS[self.experts](self._gp, self.embedding_dim)
        self._embedding_dim = self.embedding_dim
        self._model = self._expert_model
        self._model.use_multiprocessing = True
        self._expert_list.append(self._expert_model)
        self._model.build(input_shape=[[None], [None], [None]])
        if self._print_model_summary is True:
            # print model
            LOGGER.info(self._model.summary())
            LOGGER.info([(v.name, v.shape) for v in self._model.trainable_variables])

        # determine the number of steps
        num_train_triples = len(
            self._data.triples_train_idx) if self._data.triples_train_subset_idx is None else len(
            self._data.triples_train_subset_idx)
        self._steps_per_epoch = int(math.ceil(num_train_triples / self._model_params.batch_size))

        # Create dictionaries to reverse the ids to original text
        self.id2entity = {v: k for k, v in self._gp.vertex_indexer.items()}
        self.id2relation = {v: k for k, v in self._gp.relation_indexer.items()}
        # LOGGER.info('Weights of initial model: %s' % self._model.get_weights())

    def predict_set(self, triples_test: np.array, mode: str = 'tail',
                    mask_known_triples: bool = False,
                    scores_as_probabilities: bool = False) -> (np.array, list, list):
        """Predict the head, tail, or relation for samples in `triples_test`.

        Parameters
        ----------
        triples_test : np.array
            List of triples to test the model. All entities and relations have to occur in `triples_train`. "None" is
            not supported to be part of a triple. It is assumed that the passed np.array has the shape "n x 3".

        mode : {'head', 'tail', 'relation'}
            Specifies whether the model should predict the head, tail or relation of the passed list of triples

        mask_known_triples : bool
            ...

        scores_as_probabilities : bool
            ...

        Returns
        ----------
        scores : np.array
            A (n,m)-array of scores for all known entities or relations in respect of the passed `triples_test` and
            the selected `mode`. The higher the score the more likely the actual entity or relation is a correct one.
            If the normalization is set to `False` then the range of the score is [0,∞) otherwise it is [0,1]. If the
            values are normalized, a sigmoid function is applied. The order of the rows (n) corresponds to
            `triples_test` where the columns (m) are ordered by `entities`.

        entities : list
            The elements which were considered as prediction target.

        ranks : list
            The actual ranked position of the ground truth. The best position is '1'. This return value is deprecated.
        """
        if utils.is_empty(self._expert_list):
            msg = '[predict]: `_expert_list` is not initialized.'
            raise ValueError(msg)

        if utils.is_empty(triples_test):
            msg = '[predict]: `triples_test` must not be empty'
            raise ValueError(msg)

        if not utils.at_most_one_none(triples_test):
            msg = '[predict]: `triples_test` contains at least one triple where at least two elements are None.'
            raise ValueError(msg)

        # add the testing triples to `known_triples`
        known_triples = utils.get_known_triples(self._data, self._gp)

        LOGGER.info('Predicting')
        if mode == 'head' or mode == 'tail':
            nodes_idx = []
            relations_idx = []
            gold_idx = []

            triples_test = np.array([[row[0], row[1], row[0]] if row[2] is None else row for row in triples_test])
            triples_test = np.array([[row[2], row[1], row[2]] if row[0] is None else row for row in triples_test])

            for triple in triples_test:
                nodes_idx.append(self._gp.vertex_indexer[triple[0]])
                relations_idx.append(self._gp.relation_indexer[triple[1]])
                gold_idx.append(self._gp.vertex_indexer[triple[2]])

            ranks, scores = \
                self.__predict_nodes(nodes_idx, relations_idx, gold_idx, known_triples,
                                     mask_known_triples, scores_as_probabilities=scores_as_probabilities)
            entities = [self._gp.index_vertex[node_idx] for node_idx in range(self._gp.num_vertices)]
        elif mode == 'relation':
            raise NotImplementedError
        else:
            msg = '[predict]: mode needs to be head, tail or relation'
            raise ValueError(msg)

        return scores, entities, ranks

    def __get_model_params(self) -> dict:
        model_params = {'num_negative': str(self._model_params.num_negative),
                        'batch_size': str(self._model_params.batch_size),
                        'epochs': str(self._model_params.epochs),
                        'validation_step': str(self._model_params.validation_step),
                        'optimizer': str(self._model_params.optimizer),
                        'loss': str(self._model_params.loss),
                        'activation_fn': str(self._model_params.activation_fn),
                        'normalize_score': str(self._normalize_score),
                        'embedding_dim': str(self._embedding_dim),
                        'experts': self._model_params.expert,
                        'optimizer_lr': str(self._model_params.optimzer_lr),
                        'seed': str(self._model_params.seed),
                        'train_with_softmax': str(self._model_params.train_with_softmax)
                        }
        if ' ' in model_params['optimizer']:
            model_params['optimizer'] = model_params['optimizer'][
                                        1:model_params['optimizer'].find(' ')]
        for expert in self._expert_list:
            # Note: the experts need to ensure unique keys,
            # e.g. by adding their name at the beginning of the key
            model_params.update(expert.get_model_params())
        return model_params

    def save_model(self, directory_path: str, epoch: int, prefix: str = '') -> str or None:
        """Method to save the best performing model. The model is selected based on the passed validation dataset.
        In case that no validation was performed, the last model is saved.

        Parameters
        ----------
        directory_path : str
            A directory path where the model should be stored.

        Returns
        ----------
        model_path : str or None
            If it was successfully stored, the absolute file path of the model is returned otherwise None.
        prefix : str
            Name for the outputfile, if not supplied a time stamp will be used as the name instead.
        """

        # check write permission
        if not os.access(directory_path, os.W_OK):
            LOGGER.info('[save_model] Missing write permission for the passed directory path')
            return None

        if not os.path.exists(os.path.join(directory_path)):
            os.makedirs(directory_path)

        # ascertain model parameters
        model_params = self.__get_model_params()

        # save the model
        model = self._model

        if prefix == '':
            # we use the current time as unique identifier
            timestamp = str(int(time.time()))
            model_name = 'model_epoch' + epoch + '_' + timestamp
        else:
            model_name = prefix

        weights_path = os.path.join(directory_path, model_name + '.weights')
        model.save_weights(weights_path)

        # save the params
        path_params = os.path.join(directory_path, model_name + '.params')
        write_json_to_file(model_params, path_params)

        LOGGER.info('Saved the model\'s weights of the '+str(epoch)+'th epoch under %s.' % weights_path)

        return weights_path

    def load_model(self, model_path: str) -> bool:
        """
        Method to load a previously stored model.

        :param model_path: A path to a file which was stored with `save_model`
        :return: whether loading the model was successful
        """
        params_path = model_path.replace('.model', '.params')
        if not os.path.isfile(params_path):
            return False

        weight_path = model_path.replace('.model', '.weights')
        self._model.load_weights(weight_path)

        return True

    def __load_graph(self, triples_train: np.array, triples_validation: np.array,
                     triples_train_subset: np.array = None):
        if utils.is_empty(triples_train):
            msg = '[warning]: `triples_train` must not be empty'
            raise ValueError(msg)

        if utils.is_empty(triples_train):
            msg = '[warning]: `triples_validation` must not be empty'
            raise ValueError(msg)

        if not utils.uses_only_subset_of_elements(triples_train, triples_validation):
            msg = '[warning]: `triples_validation` may only contain nodes and relations which occur in `triples_train`'
            raise ValueError(msg)

        # build graph
        self.__build_graph(triples_train)

        # convert training triples to training_idx triples
        triples_train_idx = [[self._gp.vertex_indexer[triple[0]],
                              self._gp.relation_indexer[triple[1]],
                              self._gp.vertex_indexer[triple[2]]]
                             for triple in triples_train]

        triples_train_subset_idx = None
        if triples_train_subset is not None:
            triples_train_subset_idx = [[self._gp.vertex_indexer[triple[0]],
                                         self._gp.relation_indexer[triple[1]],
                                         self._gp.vertex_indexer[triple[2]]]
                                        for triple in triples_train_subset]

        # build data structure
        self._data = Data(triples_train=triples_train,
                          triples_train_idx=triples_train_idx,
                          triples_train_subset=triples_train_subset,
                          triples_train_subset_idx=triples_train_subset_idx,
                          triples_validation=triples_validation)

    def __build_graph(self, triples_train):
        nodes = list(set(triples_train[:, 0]).union(triples_train[:, 2]))
        nodes.sort()
        node_indexer = {node: idx for idx, node in enumerate(nodes)}

        relations = list(set(triples_train[:, 1]))
        relations.sort()
        relation_indexer = {rel: idx for idx, rel in enumerate(relations)}

        num_nodes = len(node_indexer)
        num_relations = len(relation_indexer)
        index_node = {idx: vertex for vertex, idx in node_indexer.items()}
        index_relation = {idx: relation for relation, idx in relation_indexer.items()}

        self._gp = Graph(vertices=nodes,
                         vertex_indexer=node_indexer,
                         num_vertices=num_nodes,
                         index_vertex=index_node,
                         relations=relations,
                         relation_indexer=relation_indexer,
                         index_relation=index_relation,
                         num_relations=num_relations)

    @tf.function
    def get_gradient(self, heads_idx: np.array, relations_idx: np.array, tails_idx: np.array):
        """
        Gets gradients for the given triples.

        :param heads_idx: All head triples, shape [Batch]
        :param relations_idx: All relation triples, shape [Batch]
        :param tails_idx: For each headxrelation, all tails to look up [Batch, x],
                x can be one for looking up 1 gold tail
        :return: all gradients
        """
        with tf.GradientTape() as tape:
            scores = self._model((heads_idx, relations_idx, tails_idx))
        grads = tape.gradient(scores, self._model.trainable_variables)
        return grads

    @tf.function
    def get_loss_and_apply_gradient(self, inputs, compute_influence_map):
        """

        :param inputs:
        :param compute_influence_map:
        :return:
        """
        heads_idx, relations_idx, pos_neg_tails, mask = inputs

        with tf.GradientTape() as tape:
            # For each entry in batch, get logits over entire nodes
            # dense_pos_neg_tails = tf.sparse.to_dense(pos_neg_tails, validate_indices=False)
            logits, norm_tuple = self._model((heads_idx, relations_idx, pos_neg_tails))
            if self.train_with_softmax is True:
                LOGGER.info('Using softmax_cross_entropy_with_logits')
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=mask, logits=logits)
            else:
                LOGGER.info('Using sigmoid_cross_entropy_with_logits')
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=logits)
            reduced_loss = tf.reduce_mean(loss)

        self._model_params.optimizer.lr.assign(self.decay_lr(self.global_step))
        grads = tape.gradient(reduced_loss, self._model.trainable_variables)
        self._model_params.optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        return reduced_loss, grads, norm_tuple

    def train_epoch(self, compute_influence_map):
        """

        :param compute_influence_map:
        :return:
        """
        step = 0
        start = time.time()
        data_generator = self.generator()
        step_counter = -1
        delete_idx_counter = 0
        for ((heads_idx, relations_idx), (pos_neg_tails, tails_idx, mask)) in data_generator:
            step_counter += 1
            if self.delete_idx is not None and delete_idx_counter < len(self.delete_idx):  # this assumes that delete_idx is sorted
                if step_counter == self.delete_idx[delete_idx_counter]:
                    delete_idx_counter += 1
                    self.global_step += 1  # this effects the decay lr. should we have it here or not?
                    # LOGGER.info('Skipping instance: %s' % step_counter)
                    continue

            prev_weights = copy.deepcopy(self._model.trainable_variables)
            loss, grads, norm_tuple = self.get_loss_and_apply_gradient((heads_idx, relations_idx, pos_neg_tails, mask), compute_influence_map)
            new_weights = self._model.trainable_variables
            infl_head, infl_rel, infl_tail = None, None, None
            if self._model_params.expert[0] == "DistMult":
                for idx, _ in enumerate(new_weights):
                    if new_weights[idx].name.startswith('distmult_node_embedding'):
                        infl_head = new_weights[idx][heads_idx[0]] - prev_weights[idx][heads_idx[0]]
                        infl_tail = new_weights[idx][pos_neg_tails[0]] - prev_weights[idx][pos_neg_tails[0]]
                    elif new_weights[idx].name.startswith('distmult_relation_embedding'):
                        infl_rel = new_weights[idx][relations_idx[0]] - prev_weights[idx][ relations_idx[0]]
            elif self._model_params.expert[0] == "ComplEx":
                infl_head_real, infl_rel_real, infl_tail_real = None, None, None
                infl_head_img, infl_rel_img, infl_tail_img = None, None, None
                for idx, _ in enumerate(new_weights):
                    if new_weights[idx].name.startswith('complex_entity_real'):
                        infl_head_real = new_weights[idx][heads_idx[0]] - prev_weights[idx][heads_idx[0]]
                        infl_tail_real = new_weights[idx][pos_neg_tails[0]] - prev_weights[idx][pos_neg_tails[0]]
                    elif new_weights[idx].name.startswith('complex_relation_real'):
                        infl_rel_real = new_weights[idx][relations_idx[0]] - prev_weights[idx][ relations_idx[0]]
                    elif new_weights[idx].name.startswith('complex_entity_img'):
                        infl_head_img = new_weights[idx][heads_idx[0]] - prev_weights[idx][heads_idx[0]]
                        infl_tail_img = new_weights[idx][pos_neg_tails[0]] - prev_weights[idx][pos_neg_tails[0]]
                    elif new_weights[idx].name.startswith('complex_relation_img'):
                        infl_rel_img = new_weights[idx][relations_idx[0]] - prev_weights[idx][ relations_idx[0]]
                infl_head = tf.concat([infl_head_real, infl_head_img], axis=0)
                infl_rel = tf.concat([infl_rel_real, infl_rel_img], axis=0)
                infl_tail = tf.concat([infl_tail_real, infl_tail_img], axis=0)
            influences = (infl_head, infl_rel, infl_tail)

            if compute_influence_map is True:
                self._model.update_bound_statistics(norm_tuple)
                self._model.write_influence_map_gradient(influences,
                                                         heads_idx.numpy().tolist(),
                                                         relations_idx.numpy().tolist(),
                                                         tails_idx.numpy().tolist(),
                                                         self.id2entity, self.id2relation)

            if step % 10000 == 0:
                end = time.time()
                LOGGER.info('Duration: %s (min)' % str((end-start)/60.0))
                start = time.time()
                LOGGER.info('Loss: %s' % loss.numpy())
                LOGGER.info('Step: %s' % step)

            step += 1
            self.global_step += 1

    def train(self, save_model: bool = True, prefix: str = '', compute_influence_map: bool = False):
        """

        :param save_model:
        :param prefix:
        :param compute_influence_map:
        :return:
        """
        known_triples = utils.get_known_triples(self._data, self._gp)

        best_mrr = -1
        self.global_step = 0

        for epoch in range(self._model_params.epochs):
            self._current_epoch = epoch
            LOGGER.info('Epoch %s/%s' % ((epoch + 1), self._model_params.epochs))
            self.train_epoch(compute_influence_map)

            if (epoch + 1) % self._model_params.validation_step == 0:
                start = time.time()

                relations_idx = list()
                nodes_idx = list()
                gold_idx = list()
                for triple in self._data.triples_validation:
                    head_idx_valid = self._gp.vertex_indexer[triple[0]]
                    relation_idx_valid = self._gp.relation_indexer[triple[1]]
                    tail_idx_valid = self._gp.vertex_indexer[triple[2]]

                    nodes_idx.append(head_idx_valid)
                    relations_idx.append(relation_idx_valid)
                    gold_idx.append(tail_idx_valid)

                LOGGER.info('Validation')
                ranks, _ = self.__predict_nodes(nodes_idx, relations_idx, gold_idx, known_triples, mask_known_triples=False)

                end = time.time()
                LOGGER.info('> Duration [min]: ' + '{:.3f}'.format((end - start) / 60))

                mrr = utils.print_model_performance(ranks)
                LOGGER.info('MRR: %s' % mrr)
                if mrr > best_mrr:
                    LOGGER.info('New best MRR: %s, previous: %s' % (mrr, best_mrr))
                    best_mrr = mrr
                    if save_model is True:
                        if self._output_dir == '':
                            raise ValueError('Make sure to set an output directory if a model should be saved.')
                        self.save_model(self._output_dir, epoch, prefix)
                        LOGGER.info('Saving model at epoch %s' % (epoch+1))
                        self._best_model_save = True
                    if self._output_dir != '' and compute_influence_map is True:
                        LOGGER.info('Writing influence file at epoch %s' % (epoch+1))
                        output_file = os.path.join(self._output_dir, "influence_map_gradient")
                        write_json_to_file(self._model.get_influence_map_gradient(), output_file)

        if self._best_model_save is False:
            if save_model is True:
                if self._output_dir == '':
                    raise ValueError(
                        'Make sure to set an output directory if a model should be saved.')
                self.save_model(self._output_dir, -1, prefix)
                LOGGER.info('Saving last model')
            if self._output_dir != '' and compute_influence_map is True:
                LOGGER.info('Saving influence file at last step')
                output_file = os.path.join(self._output_dir, "influence_map_gradient")
                write_json_to_file(self._model.get_influence_map_gradient(), output_file)
        else:
            self.load_model(os.path.join(self._output_dir, prefix+'.model'))
        # LOGGER.info('Weights of best model: %s' % self._model.get_weights())

    def __predict_nodes(self, nodes_idx: list, relations_idx: list, gold_idx: list, known_triples: set,
                        mask_known_triples: bool = False, scores_as_probabilities: bool = False) -> (list, np.array):
        # Hint: We use 'np.RandomState' as 'np.random.randint' is not thread-safe
        np_random = RandomState(0)
        nodes_idx = np.array(nodes_idx)
        relations_idx = np.array(relations_idx)

        # _keep_k_first is implemented for movielens, where the first 5 are rating and all others are users
        # we can be sure that the order stayts the same because when we build the vertex indexer, we sort
        # the nodes first and ratings have the shape 'r:*' and are thus always sorted before users, 'u:*'
        if self._keep_k_first is not None:
            scores = np.zeros((len(nodes_idx), self._keep_k_first))
        else:
            scores = np.zeros((len(nodes_idx), self._gp.num_vertices))
        for expert in self._expert_list:
            expert_score = expert.predict_tail((nodes_idx, relations_idx))
            if self._normalize_score:
                LOGGER.info('>> Normalize Scores...')
                expert_score = utils.sigmoid(expert_score)
            if self._keep_k_first is not None:
                expert_score = expert_score[:, :self._keep_k_first]
            scores += expert_score.numpy()

        if scores_as_probabilities is True:
            scores = utils.softmax(scores, axis=1)

        # the part below is quite optimized
        ranks = []
        num_scores = len(scores)
        for idx, row in tqdm.tqdm(enumerate(scores), total=num_scores, desc='> Compute Ranking'):
            node_idx_given = nodes_idx[idx]
            relation_idx_given = relations_idx[idx]
            gold_idx_given = gold_idx[idx]
            threshold_lower = row[gold_idx_given]
            rank_cleaned = 1
            equal_score = 0

            for node_idx, score in enumerate(row):
                if mask_known_triples and (node_idx_given, relation_idx_given, node_idx) in known_triples:
                    scores[idx][node_idx] = -math.inf

                if score < threshold_lower:
                    continue
                elif score == threshold_lower:
                    equal_score += 1
                elif not (node_idx_given, relation_idx_given, node_idx) in known_triples:
                    rank_cleaned += 1

            # we randomize the rank position in case several nodes have the same score as the ground truth
            if equal_score > 0:
                rank_cleaned = np_random.randint(rank_cleaned, rank_cleaned + equal_score)

            ranks.append(rank_cleaned)
        return ranks, scores

    def generator(self):
        """

        :return:
        """
        triples_train_idx = self._data.triples_train_idx if self._data.triples_train_subset_idx is None else self._data.triples_train_subset_idx
        for (idx, (head, relation, gold_tail)) in enumerate(triples_train_idx):
            # Hint: We use 'np.RandomState' as 'np.random.randint' is not thread-safe
            cou_inter = idx % 4096
            if cou_inter == 0:
                np_random = RandomState(idx)
                tail_idx_negatives = np_random.randint(self._gp.num_vertices, size=(4096, self._model_params.num_negative))

            def generate_negative_samples(head, gold_tail, tail_idx_negatives):
                replacement = int((head + gold_tail) / 2)
                neg_tail_replacement = replacement + 1 if replacement < (
                            self._gp.num_vertices - 1) else replacement

                tail_idx_negatives = np.where(tail_idx_negatives == gold_tail, neg_tail_replacement,
                                              tail_idx_negatives)
                tail_idx_negatives = np.insert(tail_idx_negatives, 0, gold_tail)
                return tail_idx_negatives

            pos_neg_tails = generate_negative_samples(head, gold_tail, tail_idx_negatives[cou_inter])
            mask = np.array([1.] + [0.] * (len(pos_neg_tails)-1), dtype=self.np_precision)

            device = '/gpu:0'
            if self._deterministic is True:
                device = '/cpu:0'
            with tf.device(device):
                a = tf.constant(np.array([head]))
                b = tf.constant(np.array([relation]))
                c = tf.constant(np.array(pos_neg_tails))
                d = tf.constant(np.array([gold_tail]))
                e = tf.constant(np.array([mask]))

            yield (a, b), (c, d, e)
