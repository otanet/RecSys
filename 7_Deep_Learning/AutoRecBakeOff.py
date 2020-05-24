# -*- coding: utf-8 -*-

from MovieLens import MovieLens
from AutoRecAlgorithm import AutoRecAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("映画の評価点数の算出中...")
    data = ml.loadMovieLensLatestSmall()
    print("\n新規性の測定ができるように、映画の人気度の硬いランキングを計算...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# 基本的なデータセットをロードする
(ml, evaluationData, rankings) = LoadMovieLensData()

# 評価
evaluator = Evaluator(evaluationData, rankings)

#Autoencoder
AutoRec = AutoRecAlgorithm()
evaluator.AddAlgorithm(AutoRec, "AutoRec")

# ランダムでのレコメンド
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# 評価測定
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
