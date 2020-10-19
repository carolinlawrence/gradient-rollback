# coding: utf-8
#        Gradient Rollback
#
#   File:     delete_facts.py
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
Given a set of triples, provides different methods how some triples (facts) are deleted.
"""

import logging
import numpy as np
from argparse import Namespace

from gr.utils.read_write import read_json, read_lines_in_list

LOGGER = logging.getLogger(__name__)


class FactDeleter:
    """
    Superclass for all fact deleters.
    """
    def __init__(self):
        pass

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def remove_triples(self, set_of_triplets, nr_explanations: int = 0):
        """
        Given a set of triplets, remove some triplets
        :param set_of_triplets: a set of triplets
        :param nr_explanations: used by NH
        :return: a list of the indices that should be deleted
        """
        raise NotImplementedError

    def set_triple_of_interest(self, triple_of_interest) -> int:
        """
        Sets the current triple.
        :param triple_of_interest: the current triple, can be transformed into the format
        that the deleter needs
        :return: 0 on success
        """
        raise NotImplementedError

    def print_statistics(self) -> int:
        return 0


class NHDeleter(FactDeleter):
    """
    Given a fact delete at random from the set of adjacent training triples (i.e. for a triple of
    interest (h, r, t), the set off adjacent training triples are all training triples that contain
    at least one h/r/t.
    """
    def __init__(self, delete_amount: int = -1):
        """
        :param delete_amount: how many to delete
        :param explanations_file_path: contains the triples we should train an auxiliarly models for
                (at the moment assumed to be top1_correct_predictions.txt as return from training
                a main model, has the format per line: head\trel\ttail)
        """
        super().__init__()
        self._triple_of_interest = None
        self._delete_amount = delete_amount

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def remove_triples(self, triples: np.array, nr_explanations: int = 0) -> np.array:
        """
        Given a set of triplets, remove some triplets at random
        :param set_of_triplets: a set of triplets
        :param nr_explanations: the number of explanations to delete
        :return: a list of the indices that should be deleted
        """
        self._deletion_vector = np.zeros(len(triples))
        delete_idx = []
        collect_triples = []
        for idx, triple in enumerate(triples):
            if_passed = False
            if triple[0] == self._triple_of_interest[0]:
                if_passed = True
            if triple[1] == self._triple_of_interest[1]:
                if_passed = True
            if triple[2] == self._triple_of_interest[2]:
                if_passed = True
            if triple[2] == self._triple_of_interest[0]:
                if_passed = True
            if triple[0] == self._triple_of_interest[2]:
                if_passed = True
            if if_passed is True:
                delete_idx.append(idx)
                collect_triples.append("\t".join(triple))
        # delete the lines and mark the delete rows in our mask vector (1==deleted)
        self._deletion_vector[delete_idx] = 1
        if self._delete_amount > 0 and len(delete_idx) > self._delete_amount:
            delete_idx = np.random.choice(delete_idx, self._delete_amount, replace=False)
        if self._delete_amount == -1:
            delete_idx = np.random.choice(delete_idx, nr_explanations, replace=False)
        LOGGER.info('Triples deleted: %s' % len(delete_idx))
        return delete_idx

    def set_triple_of_interest(self, triple_of_interest) -> int:
        self._triple_of_interest = triple_of_interest
        return 0


class ExplanationDeleter(FactDeleter):
    def __init__(self, explanation_file_path):
        super().__init__()
        self.explanation_file_path = explanation_file_path
        self._deletion_vector = None
        self._triple_of_interest = None

        self._read_file()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def remove_triples(self, triples: np.array, nr_explanations: int = 0) -> np.array:
        """
        Given a set of triplets, remove some triplets depending on the relation
        :param triples: a list of triplets
        :param nr_explanations: used by NH
        :return: a list of the indices that should be deleted
        """

        # we create a one dimensional vector for marking triples which were deleted
        self._deletion_vector = np.zeros(len(triples))

        # in case that there is no entry for the triple of interest we just do nothing
        if self._triple_of_interest not in self._filter:
            return triples

        # create a list of triples which we want to delete (explanations)
        explanations = []
        for entry in self._filter[self._triple_of_interest]['explanation']:
            explanations += ['_'.join(entry[0])]

        # let's build an index for faster look-ups
        # the keys are the triples of the passed np.array, the values are the corresponding line
        triples_dict = {}
        for idx, row in enumerate(triples):
            key = '_'.join(row)
            if key not in triples_dict:
                triples_dict[key] = []
            triples_dict[key].append(idx)  # It might be possible that the same triple occurs several times

        # now identify the lines which we want to delete (we build a list)
        delete_idx = np.array([triples_dict[entry] for entry in explanations if entry in triples_dict])
        delete_idx = delete_idx.flatten()

        # delete the lines and mark the delete rows in our mask vector (1==deleted)
        self._deletion_vector[delete_idx] = 1
        return delete_idx

    def set_triple_of_interest(self, triple_of_interest) -> int:
        self._triple_of_interest = ' '.join(triple_of_interest)
        return 0

    def _read_file(self):
        self._filter = read_json(self.explanation_file_path)


def get_fact_deleter(prog_args: Namespace, training_set_size: int, number_of_triples: int) -> FactDeleter:
    """
    Factory for returning a fact deleter
    (subclass instance of :py:class:FactDeleter from delete_facts.py)
    :param prog_args: instance of :py:class:ArgumentParser
    :return: a subclass instance of :py:class:FactDeleter
    """
    if prog_args.delete_facts == 'GR':
        fact_deleter = ExplanationDeleter(prog_args.explanations_file_path)
    elif prog_args.delete_facts == 'NH':
        fact_deleter = NHDeleter(prog_args.delete_amount)
    else:
        raise NotImplementedError
    return fact_deleter
