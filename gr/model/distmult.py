# coding: utf-8
#        Gradient Rollback
#
#   File:     distmult.py
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
Implements DistMult
"""

import tensorflow as tf
import numpy as np
import logging

from collections import namedtuple
from collections import defaultdict

from gr.model.expert import Expert

LOGGER = logging.getLogger(__name__)

try:
    from gr.model.custom_embeddings import CustomEmbedding
except Exception as e:
    LOGGER.warning('Please see "Dependencies" in README.md to either (1) modify the TF Embedding layer or (2) comment this import and uncomment the one below. Note that option (2) will drastically slow down training and the evaluation step would take a very long time.')
    raise e
# from tensorflow.keras.layers import Embedding


class DistMult(Expert):
    """
    Implements DistMult
    """
    def __init__(self, graph_properties: namedtuple, embedding_dim: int,
                 compute_influence_map: bool = False,
                 batch_larger_one: bool = False):
        """
        Set up the required variables.
        :param graph_properties: Accessed to derive the number of entities and relations
        :param embedding_dim: The embedding dimension size.
        """
        super().__init__()
        self._name = 'distmult_'
        self._embedding_dim = embedding_dim
        self._compute_influence_map = compute_influence_map
        self.max_constant = 0.0
        self.tf_precision = tf.float64

        self._node_embedding = CustomEmbedding(input_dim=graph_properties.num_vertices,
                                         output_dim=self._embedding_dim,
                                         embeddings_initializer=tf.initializers.RandomUniform(),
                                         name=self._name + 'node_embedding')
        self._relation_embedding = CustomEmbedding(input_dim=graph_properties.num_relations,
                                             output_dim=self._embedding_dim,
                                             embeddings_initializer=tf.initializers.RandomUniform(),
                                             name=self._name + 'relation_embedding')
        # Keeps track of influence during step 1.
        self._influence_map_gradients = defaultdict(float)
        # The below variables are used for step 2.
        self._previous_node_weights = None
        self._previous_relation_weights = None
        self._batch_larger_one = batch_larger_one

    @tf.function
    def call(self, inputs):
        """
        Given inputs, compute the DistMult scoring function for the indicates tails.
        :param inputs: a tuple of lists containing head/relation/tail indices.
        :return: For each head/relation/trail, the computed score
        """
        heads_idx, relations_idx, tails_idx = inputs
        if self._batch_larger_one:
            # expand_dims needed for matmult to work correctly
            head_embedded = self._node_embedding(tf.expand_dims(heads_idx, 1))  # Batch x 1 x Hidden
            relation_embedded = self._relation_embedding(
                tf.expand_dims(relations_idx, 1))  # Batch x 1 x Hidden
            tail_embedded = self._node_embedding(tails_idx)  # Batch x (num_neg+1) x Hidden
            scores_emb = tf.matmul(head_embedded * relation_embedded, tail_embedded, transpose_b=True)  # Batch x 1 x  (num_neg+1)
            scores_emb = tf.squeeze(scores_emb, axis=1)  # Batch x (num_neg+1)
        else:
            head_embedded = self._node_embedding(heads_idx)
            relation_embedded = self._relation_embedding(relations_idx)
            tail_embedded = self._node_embedding(tails_idx)
            head_rel = head_embedded*relation_embedded
            scores_emb = tf.matmul(head_rel, tail_embedded, transpose_b=True)
        return scores_emb

    def update_bound_statistics(self, inputs):
        """
        Given (head, relation, gold_tail), look up embeddings and compute the norm for each pair
        multitplied. Keep track of the maxmimum over entirety of training to find the Lipschitz
        constant. See also Lemma 1 & 2 in the paper.
        :param inputs: a tuple of lists containing head/relation/gold tail indices.
        :return: 0 on success (update the internal maximum trakcer self.max_constant)
        """
        head_rel, rel_tail, head_tail = inputs
        self.max_constant = np.max([head_rel, rel_tail, head_tail, self.max_constant])
        return 0

    def print_bound_statistics(self, output_dir, epoch: int = None):
        if epoch is None:
            info_string = 'at the last step'
        else:
            info_string = 'at epoch %s' % (epoch+1)
        LOGGER.info('The Lipschitz constant %s is: %s' % (info_string, self.max_constant))

    @tf.function
    def predict_tail(self, inputs):
        """
        Given inputs, compute the DistMult scoring function for all tails.
        :param inputs: a tuple of lists containing head/relation indices.
        :return: For each head/relation pair, the scores over all tails
        """
        heads_idx, relations_idx = inputs
        head_embedded = self._node_embedding(heads_idx)  # Batch x Hidden
        relation_embedded = self._relation_embedding(relations_idx)  # Batch x Hidden
        scores_emb = tf.matmul(head_embedded * relation_embedded, self._node_embedding.embeddings, transpose_b=True)  # Batch x |Nodes|
        return scores_emb

    def update_according_to_influence(self, indices, influences):
        """
        Resets the weights according to the influence passed. Used by step 2.
        :param indices: tuple of which head/rel/tail index is effected
        :param influences: tuple of influence of head/rel/tail
        :return: 0 on success
        """
        heads_idx, relations_idx, tails_idx = indices
        head_influence, relation_influence, tail_influence = influences
        self._previous_node_weights = self.get_weights()[0]
        self._previous_relation_weights = self.get_weights()[1]
        modify_weights_node = self.get_weights()[0]
        modify_weights_rel = self.get_weights()[1]
        modify_weights_node[heads_idx] = modify_weights_node[heads_idx] - head_influence
        modify_weights_rel[relations_idx] = modify_weights_rel[relations_idx] - relation_influence
        modify_weights_node[tails_idx] = modify_weights_node[tails_idx] - tail_influence
        self.set_weights([modify_weights_node, modify_weights_rel])
        return 0

    def revert_weights(self):
        """
        Revert to original weights, undoes the function update_according_to_influence. Used by step 2.
        Assumes update_according_to_influence is called first, else the variables are None.
        :return: 0 on success
        """
        self.set_weights([self._previous_node_weights, self._previous_relation_weights])
        return 0
