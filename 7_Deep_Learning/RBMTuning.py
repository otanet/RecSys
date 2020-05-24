# -*- coding: utf-8 -*-

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("映画の評価値のロード中...")
    data = ml.loadMovieLensLatestSmall()
    print("\n後で新規性を測定できるように、映画の人気の高いランクを計算中...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

#レコメンドアルゴリズムの一般的なデータセットをロードする
(ml, evaluationData, rankings) = LoadMovieLensData()

print("最も良いパラメーターを探しています...")
param_grid = {'hiddenDim': [20, 10], 'learningRate': [0.1, 0.01]}
gs = GridSearchCV(RBMAlgorithm, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)

#最も良いRMSEもスコア
print("最も良いRMSEのスコアが得られました: ", gs.best_score['rmse'])

#最も良いRMSEのスコアが与えられたパラメーターの組み合わせを表示する
print(gs.best_params['rmse'])

#「Evaluator」を構築して、評価する
evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
RBMtuned = RBMAlgorithm(hiddenDim = params['hiddenDim'], learningRate = params['learningRate'])
evaluator.AddAlgorithm(RBMtuned, "RBM - Tuned")

RBMUntuned = RBMAlgorithm()
evaluator.AddAlgorithm(RBMUntuned, "RBM - Untuned")

#無作為にレコメンドする
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

#両者を比較考察する
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
