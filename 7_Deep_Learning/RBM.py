# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class RBM(object):

    def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50, ratingValues=10, learningRate=0.001, batchSize=100):

        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
        self.batchSize = batchSize
        
                
    def Train(self, X):

        for epoch in range(self.epochs):
            np.random.shuffle(X)
            
            trX = np.array(X)
            for i in range(0, trX.shape[0], self.batchSize):
                epochX = trX[i:i+self.batchSize]
                self.MakeGraph(epochX)

            print("Trained epoch ", epoch)


    def GetRecommendations(self, inputUser):
        
        feed = self.MakeHidden(inputUser)
        rec = self.MakeVisible(feed)
        return rec[0]       

    def MakeGraph(self, inputUser):

        #ランダムに重みづけを初期化する
        maxWeight = -4.0 * np.sqrt(6.0 / (self.hiddenDimensions + self.visibleDimensions))
        self.weights = tf.Variable(tf.random.uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), tf.float32, name="weights")
        
        self.hiddenBias = tf.Variable(tf.zeros([self.hiddenDimensions], tf.float32, name="hiddenBias"))
        self.visibleBias = tf.Variable(tf.zeros([self.visibleDimensions], tf.float32, name="visibleBias"))
        
        #k=1として、コンストラクティブ・ダイバージェンス法でギブス・サンプリングする
        #適切になるまで複数回試行する
        
        # 前向きなフォーワードで行う
        # 隠れ層からサンプリングする
        # 隠れ層の確率のテンソルを取得する
        hProb0 = tf.nn.sigmoid(tf.matmul(inputUser, self.weights) + self.hiddenBias)
        # すべての分布からサンプリングする
        hSample = tf.nn.relu(tf.sign(hProb0 - tf.random.uniform(tf.shape(hProb0))))
        #転置する
        forward = tf.matmul(tf.transpose(inputUser), hSample)
        
        #後ろ向きの推論を行う
        # 可視化にむけて、隠れ層にサンプルが与えられた層を再構築する
        v = tf.matmul(hSample, tf.transpose(self.weights)) + self.visibleBias
        
        #欠損値にマスクする
        vMask = tf.sign(inputUser) # Make sure everything is 0 or 1
        vMask3D = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues]) # Reshape into arrays of individual ratings
        vMask3D = tf.reduce_max(vMask3D, axis=[2], keepdims=True) # Use reduce_max to either give us 1 for ratings that exist, and 0 for missing ratings
        
        #個人別の10個の評価値のあるバイナリーから評価値のベクトルを抽出する
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.ratingValues])
        vProb = tf.nn.softmax(v * vMask3D) # Apply softmax activation function
        vProb = tf.reshape(vProb, [tf.shape(v)[0], -1]) # And shove them back into the flattened state. Reconstruction is done now.
        #後ろ向き推論と隠れ層のバイアスを定義し、結合する
        hProb1 = tf.nn.sigmoid(tf.matmul(vProb, self.weights) + self.hiddenBias)
        backward = tf.matmul(tf.transpose(vProb), hProb1)
    
        #エポックごとに何が行われたかを定義する
        #前向き推論と後ろ向き推論の処理を終えたら、重みづけの値を更新する
        weightUpdate = self.weights.assign_add(self.learningRate * (forward - backward))
        #隠れ層のダイバージェンスを最小化して、隠れ層のバイアスを更新する
        hiddenBiasUpdate = self.hiddenBias.assign_add(self.learningRate * tf.reduce_mean(hProb0 - hProb1, 0))
        #結果を可視化するために、ダイバージェンスの最小化をしつつ、可視化できるバイアスを更新する
        visibleBiasUpdate = self.visibleBias.assign_add(self.learningRate * tf.reduce_mean(inputUser - vProb, 0))

        self.update = [weightUpdate, hiddenBiasUpdate, visibleBiasUpdate]
        
    def MakeHidden(self, inputUser):
        hidden = tf.nn.sigmoid(tf.matmul(inputUser, self.weights) + self.hiddenBias)
        self.MakeGraph(inputUser)
        return hidden
    
    def MakeVisible(self, feed):
        visible = tf.nn.sigmoid(tf.matmul(feed, tf.transpose(self.weights)) + self.visibleBias)
        #self.MakeGraph(feed)
        return visible
