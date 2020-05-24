# -*- coding: utf-8 -*-

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("映画の評価値の算出中...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

#レコメンドのアルゴリズムに対して、一般的なデータをロードする。Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

#「Evaluator」クラスを作り、評価する。
evaluator = Evaluator(evaluationData, rankings)

#RBMでエポックを20とする。
RBM = RBMAlgorithm(epochs=20)
evaluator.AddAlgorithm(RBM, "RBM")

#無作為にレコメンデーションする
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

#両者を比較する
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
