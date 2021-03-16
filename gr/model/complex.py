# coding: utf-8
#        Gradient Rollback
#
#   File:     complex.py
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


class ComplEx(Expert):
    """
    Implements ComplEx
    """
    def __init__(self, graph_properties: namedtuple, embedding_dim: int,
                 compute_influence_map: bool = False,
                 regularizer_name: str = '', regularizer_strength: float = 0.005,
                 batch_larger_one: bool = False):
        super().__init__()
        self._name = 'complex_'
        self._embedding_dim = embedding_dim
        self._regularizer_name = regularizer_name
        self._regularizer_strength = regularizer_strength
        self._compute_influence_map = compute_influence_map
        self._influence_map_gradients = defaultdict(float)
        self.embed_e_real = CustomEmbedding(input_dim=graph_properties.num_vertices,
                                      output_dim=self._embedding_dim,
                                      embeddings_initializer=tf.initializers.RandomUniform(),
                                      name=self._name + 'entity_real')

        self.embed_e_img = CustomEmbedding(input_dim=graph_properties.num_vertices,
                                     output_dim=self._embedding_dim,
                                     embeddings_initializer=tf.initializers.RandomUniform(),
                                     name=self._name + 'entity_img')

        self.embed_rel_real = CustomEmbedding(input_dim=graph_properties.num_relations,
                                        output_dim=self._embedding_dim,
                                        embeddings_initializer=tf.initializers.RandomUniform(),
                                        name=self._name + 'relation_real')

        self.embed_rel_img = CustomEmbedding(input_dim=graph_properties.num_relations,
                                       output_dim=self._embedding_dim,
                                       embeddings_initializer=tf.initializers.RandomUniform(),
                                       name=self._name + 'relation_img')
        self._previous_e_real_weights = None
        self._previous_e_img_weights = None
        self._previous_rel_real_weights = None
        self._previous_rel_img_weights = None

    @tf.function
    def call(self, inputs):
        head, relation, tail = inputs

        head_real = self.embed_e_real(head)
        head_img = self.embed_e_img(head)
        relation_real = self.embed_rel_real(relation)
        relation_img = self.embed_rel_img(relation)
        tail_real = self.embed_e_real(tail)
        tail_img = self.embed_e_img(tail)

        realrealreal = tf.matmul(head_real * relation_real, tail_real, transpose_b=True)
        realimgimg = tf.matmul(head_real * relation_img, tail_img, transpose_b=True)
        imgrealimg = tf.matmul(head_img * relation_real, tail_img, transpose_b=True)
        imgimgreal = tf.matmul(head_img * relation_img, tail_real, transpose_b=True)

        scores_emb = realrealreal + realimgimg + imgrealimg - imgimgreal
        return scores_emb

    def predict_tail(self, inputs):
        head, relation = inputs

        head_real = self.embed_e_real(head)
        head_real = head_real.numpy()
        head_img = self.embed_e_img(head)
        head_img = head_img.numpy()

        relation_real = self.embed_rel_real(relation)
        relation_real = relation_real.numpy()
        relation_img = self.embed_rel_img(relation)
        relation_img = relation_img.numpy()

        tail_real = self.embed_e_real.embeddings.numpy()
        tail_img = self.embed_e_img.embeddings.numpy()

        realrealreal = np.dot(head_real * relation_real, tail_real.T)
        realimgimg = np.dot(head_real * relation_img, tail_img.T)
        imgrealimg = np.dot(head_img * relation_real, tail_img.T)
        imgimgreal = np.dot(head_img * relation_img, tail_real.T)

        scores_emb = realrealreal + realimgimg + imgrealimg - imgimgreal
        return tf.convert_to_tensor(scores_emb)

    def get_model_params(self) -> dict:
        """
        Return the emebdding dimension as a dictionary
        :return: a dictionary with the emebdding dimension as a key/value pair
        """
        model_params = {self._name+'embedding_dim': str(self._embedding_dim)}
        return model_params

    def update_according_to_influence(self, indices, influences):
        heads_idx, relations_idx, tails_idx = indices
        head_influence, relation_influence, tail_influence = influences
        self._previous_e_real_weights = self.embed_e_real.get_weights()[0]
        self._previous_e_img_weights = self.embed_e_img.get_weights()[0]
        self._previous_rel_real_weights = self.embed_rel_real.get_weights()[0]
        self._previous_rel_img_weights = self.embed_rel_img.get_weights()[0]
        modify_weights_node = self.embed_e_real.get_weights()[0]
        modify_weights_node_img = self.embed_e_img.get_weights()[0]
        modify_weights_rel = self.embed_rel_real.get_weights()[0]
        modify_weights_rel_img = self.embed_rel_img.get_weights()[0]
        #by definition of how we saved the weights the first part is real and the second img
        modify_weights_node[heads_idx] = modify_weights_node[heads_idx] - head_influence[:int(len(head_influence)/2)]
        modify_weights_node_img[heads_idx] = modify_weights_node_img[heads_idx] - head_influence[int(len(head_influence)/2):]
        modify_weights_rel[relations_idx] = modify_weights_rel[relations_idx] - relation_influence[:int(len(head_influence)/2)]
        modify_weights_rel_img[relations_idx] = modify_weights_rel_img[relations_idx] - relation_influence[int(len(head_influence)/2):]
        modify_weights_node[tails_idx] = modify_weights_node[tails_idx] - tail_influence[:int(len(head_influence)/2)]
        modify_weights_node_img[tails_idx] = modify_weights_node_img[tails_idx] - tail_influence[int(len(head_influence)/2):]
        self.embed_e_real.set_weights([modify_weights_node,])
        self.embed_e_img.set_weights([modify_weights_node_img,])
        self.embed_rel_real.set_weights([modify_weights_rel,])
        self.embed_rel_img.set_weights([modify_weights_rel_img,])
        return 0

    def revert_weights(self):
        self.embed_e_real.set_weights([self._previous_e_real_weights])
        self.embed_e_img.set_weights([self._previous_e_img_weights])
        self.embed_rel_real.set_weights([self._previous_rel_real_weights])
        self.embed_rel_img.set_weights([self._previous_rel_img_weights])
        return 0