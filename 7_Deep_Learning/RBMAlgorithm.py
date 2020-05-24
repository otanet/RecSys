# -*- coding: utf-8 -*-

from surprise import AlgoBase
from surprise import PredictionImpossible
import numpy as np
from RBM import RBM

class RBMAlgorithm(AlgoBase):

    def __init__(self, epochs=20, hiddenDim=100, learningRate=0.001, batchSize=100, sim_options={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hiddenDim = hiddenDim
        self.learningRate = learningRate
        self.batchSize = batchSize
        
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        numUsers = trainset.n_users
        numItems = trainset.n_items
        
        trainingMatrix = np.zeros([numUsers, numItems, 10], dtype=np.float32)
        
        for (uid, iid, rating) in trainset.all_ratings():
            adjustedRating = int(float(rating)*2.0) - 1
            trainingMatrix[int(uid), int(iid), adjustedRating] = 1
        
        #ユーザーごとにアイテムの可能な評価スコアに対して、ノードに2次元配列で平坦化する
        trainingMatrix = np.reshape(trainingMatrix, [trainingMatrix.shape[0], -1])
        
        #可視化できるノードにアイテムの整数値と評価値を掛け算して、RBMを新たに構築する
        rbm = RBM(trainingMatrix.shape[1], hiddenDimensions=self.hiddenDim, learningRate=self.learningRate, batchSize=self.batchSize, epochs=self.epochs)
        rbm.Train(trainingMatrix)

        self.predictedRatings = np.zeros([numUsers, numItems], dtype=np.float32)
        for uiid in range(trainset.n_users):
            if (uiid % 50 == 0):
                print("Processing user ", uiid)
            recs = rbm.GetRecommendations([trainingMatrix[uiid]])
            recs = np.reshape(recs, [numItems, 10])
            
            for itemID, rec in enumerate(recs):
                #評価値に最も高いスコアを付与して明示する                
                #rating = rec.argmax()
                #5つ星に複数の評価値を導入する
                #考えうるk値の正規化を行う
                #予測する際に、期待値を取る
                normalized = self.softmax(rec)
                rating = np.average(np.arange(10), weights=normalized)
                self.predictedRatings[uiid, itemID] = (rating + 1) * 0.5
        
        return self


    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        rating = self.predictedRatings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating
    