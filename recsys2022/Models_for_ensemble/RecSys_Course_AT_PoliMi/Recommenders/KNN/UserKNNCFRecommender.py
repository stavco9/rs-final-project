#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_utils import check_matrix
from Recommenders.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender

from Recommenders.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

from Recommenders.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCFRecommender(BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]


    def __init__(self, URM_train, verbose = True):
        super(UserKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, feature_weighting = 'none',start_user=None, **similarity_args):

        self.topK = similarity_args["topK"]
        self.shrink = similarity_args["shrink"]
        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        similarity = Compute_Similarity(self.URM_train.T, **similarity_args)

        self.W_sparse = similarity.compute_similarity(end_col=start_user)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        
    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False
        user_id_array=user_id_array.copy()-self.offset
        #print(user_id_array)
        return super(UserKNNCFRecommender, self).recommend(user_id_array,cutoff,remove_seen_flag,items_to_compute,remove_top_pop_flag,remove_custom_items_flag ,return_scores)
        
