# coding: utf-8
#        Gradient Rollback
#
#   File:     program_arguments.py
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
Handles the possible command line arguments for GR
"""

from random import randrange
from argparse import ArgumentParser


def generate_seed(prog_args):
    """
    Generates a random seed.
    :param prog_args: a ArgumentParser with a 'seed' int argument.
    :return: the ArgumentParser with a randomly set seed accessible via prog_args.seed
    """
    prog_args.seed = randrange(2 ** 32 - 1)
    return prog_args


def make_xai_parser(step) -> ArgumentParser:
    """
    returns a parser for for the different GR steps
    :param: step: Either 1, 2 or 3, reflecting on of the three steps for GR.
    :return: an ArgumentParser
    """

    def __add_arguments():
        """
        Adds all general arguments of interest.
        :return: 0 on success
        """
        # Required parameters
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where output files will be written.")
        parser.add_argument("--dataset", default=None, type=str, required=True,
                            help="The dataset to use.")
        parser.add_argument("--gpu", default='-1', type=str,
                            help="GPU which should be used")
        parser.add_argument("--deterministic", default=False, type=bool,
                            help="Force deterministic results. This increases the training duration.")
        parser.add_argument("--switch_float32", default=False, type=bool,
                            help="If true, use float32 precision.")
        return 0

    def __add_model_params_file():
        parser.add_argument("--params", default=None, type=str, required=True,
                            help="A file with the parameters with which ModelHandler should be "
                                 "initialized. This should be the parameter file of the main model"
                                 "that we want to explain.")
        return 0

    def __add_step_2_arguments():
        """
        Adds all arguments for step 2: getting explanations
        :return: 0 on success
        """
        # Required parameters
        parser.add_argument("--model", default=None, type=str, required=True,
                            help="The main model that we want to explain.")
        parser.add_argument("--influence_map", default=None, type=str, required=True,
                            help="The influence map on the basis of which explanations "
                                 "will be provided. This is an output of step 1 (except if "
                                 "--no_influence True is passed).")
        parser.add_argument("--triples", default=None, type=str, required=True,
                            help="The list of triples for which to provide explanations.")
        parser.add_argument("--correct_ids", type=str, default=None,
                            help="If set, then the values in this file are assumed to indicate"
                                 "correct lines in --predictions_file_path"
                                 "(zero-index, file is written by step one and called correct_ids,"
                                 "should only be used in conjunction with all predictions.)")
        return 0

    def __add_step_1_arguments():
        """
        Adds all arguments for step 1: training a main model
        :return: 0 on success
        """
        parser.add_argument("--epochs", default=1, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--batch_size", default=256, type=int,
                            help="The batch size for training.")
        parser.add_argument("--num_negative", default=100, type=int,
                            help="The number of negative samples to use for training.")
        parser.add_argument("--top_k", default=-1, type=int,
                            help="Return the top k predictions.")
        parser.add_argument("--optimizer", type=str, default='adam',
                            choices=['adam', 'sgd'],
                            help="Which optmizer to use")
        parser.add_argument("--learning_rate", type=float, default=0.001,
                            help="The initial learning rate for the optimizer."
                                 "(Note that we always apply a decay_lr)")
        parser.add_argument("--latent_expert_embedding_dim", default=100, type=int,
                            help="Total number of hidden embeddings to use for the latent expert.")
        parser.add_argument("--expert", default='DistMult', type=str,
                            choices=['DistMult', 'ComplEx'],
                            help="Which expert to use.")
        parser.add_argument("--validation_step", default=5, type=int,
                            help="After how many epochs a validation step should be run.")
        parser.add_argument("--seed", default=42, type=int,
                            help="Set a seed, if -1, then a random seed will be drawn.")
        parser.add_argument("--train_with_softmax", type=bool,
                            help="If true, use softmax instead of sigmoid during training.")
        parser.add_argument("--skip_in_training", default='', type=str,
                            help="A list, one entry per line, with the indices that should be "
                                 "skipped during training (zero-indexed)")
        return 0


    def __add_step_3_arguments():
        """
        Adds all arguments for step 3: evaluation
        :return: 0 on success
        """
        parser.add_argument("--delete_facts", default=None, type=str,
                            help="Which deleter to use for evaluation.",
                            choices=['GR', 'NH'])
        parser.add_argument("--explanations_file_path", default=None, type=str,
                            help="Location of the explanation file.")
        parser.add_argument("--delete_amount", default=-1, type=int,
                            help="NH deleter: delete this many instances. Setting -1 will delete as many instances "
                                 "as listed in explanation file under 'number_explanations'")
        return 0

    def __add_step_1_3_arguments():
        """
        Adds step 1/3 specific arguments.
        :return: 0 on success
        """
        # Required parameters
        parser.add_argument("--no_save", type=bool,
                            help="If passed, then models aren't saved.")
        parser.add_argument("--no_influence", type=bool,
                            help="If passed, then influence maps are calculated.")
        return 0

    parser = ArgumentParser(description='Handles the program arguments for GR: step 1, 2 and 3 respectively.')
    __add_arguments()
    if step == 1:
        __add_step_1_arguments()
        __add_step_1_3_arguments()
    if step == 2:
        __add_step_2_arguments()
        __add_model_params_file()
    if step == 3:
        __add_step_3_arguments()
        __add_step_1_3_arguments()
        __add_model_params_file()
    return parser
