
#from scipy.sparse import csgraph
#from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

#import matplotlib.pyplot as plt
#import networkx as nx
import numpy as np
#import scipy.sparse as sp

import OutlierDetector as od

class CMOD(od.OutlierDetector):

  def __init__(self,numViews):
    '''
    Constructor
    '''
    self.beta = 0.6
    self.gamma = 0.01

    super(CMOD, self).__init__(numViews)

  def getParams(self,params):

    if "k" in params:
      k = params["k"]
    else:
      k = self.k

    if "beta" in params:
      beta = params["beta"]
    else:
      beta = self.beta

    if "gamma" in params:
      gamma = params["gamma"]
    else:
      gamma = self.gamma


    return int(k),beta,gamma

  def updateM(self, G, GS):

    numViews = len(G)
    K = GS.shape[0]

    M = [np.zeros((K, K))] * numViews
    iG = np.linalg.pinv(GS)
    for i in range(numViews):
      C = np.matmul(G[i], iG)
      M[i] = np.matmul(C, np.diag(1.0 / np.sum(C, axis=0)))

      return M

  def updateGS(self, M, G, beta=1):

    numViews = len(M)
    numSamples = G[0].shape[1]
    K = M[0].shape[0]

    canSol = np.eye(K)
    GS = np.zeros((K, numSamples))

    for m in range(numSamples):
      cost = np.array([0] * K)
      for k in range(K):
        for i in range(numViews):
          cost[k] = cost[k] + beta * np.linalg.norm(G[i][:, m] - np.matmul(M[i], canSol[:, k])) ** 2

          GS[:, m] = canSol[:, np.argmin(cost)]

    # print cost

    return GS

  def updateGDMOD(self, i, X, G, M, S, H, Y, mu=1, beta=1):

    numViews = len(S)
    numSamples = S[0].shape[1]
    K = M[0].shape[0]

    canSol = np.eye(K)
    g = np.zeros((K, numSamples))
    valors = np.zeros(numSamples)
    for m in range(numSamples):
      cost = [0] * K
      for j in range(i) + range(i + 1, numViews):
        if i < j:
          pos = j * (j - 1) / 2 + i
        else:
          pos = i * (i - 1) / 2 + j
        for k in range(K):
          a = canSol[:, k] - np.matmul(M[pos], G[j][:, m])

          aux = X[i][:, m] - np.matmul(H[i], canSol[:, k])  # - S[i][:,m]

          cost[k] = cost[k] + beta * np.linalg.norm(a) + np.linalg.norm(aux)

      g[:, m] = canSol[:, np.argmin(cost)]
      valors[m] = min(cost)

      # print cost

    return g

  def updateG(self, X, H, S, M, Y, gs, mu=1, beta=1):

    numViews = len(S)
    numSamples = S[0].shape[1]
    K = M[0].shape[0]

    canSol = np.eye(K)
    G = [np.zeros((K, numSamples))] * numViews

    for m in range(numSamples):
      cost = np.zeros((K, numViews))
      for k in range(K):
        for i in range(numViews):
          a = canSol[:, k] - np.matmul(M[i], gs[:, m])

          aux = X[i][:, m] - np.matmul(H[i], canSol[:, k]) - S[i][:, m]

          cost[k, i] = beta * np.linalg.norm(a) ** 2 + np.dot(Y[i][:, m], aux) + mu / 2 * np.linalg.norm(aux) ** 2

        pos = np.argmin(cost, axis=0)
        for i in range(numViews):
          G[i][:, m] = canSol[:, pos[i]]

        # print cost

    return G

  def updateH(self, x, g, s, y, mu=1):
    return np.matmul(y + mu * (x - s), np.linalg.pinv(g)) / mu

  def updateS(self, x, h, g, y, mu=1):

    eta = 1 / mu

    t = x - np.matmul(h, g) + y / mu

    s = np.zeros(x.shape)
    for i, val in enumerate(np.linalg.norm(t, axis=0)):
      if val > eta:
        s[:, i] = (val - eta) / val * t[:, i]

    return s

  def score(self, G, S, gamma=0.01):

    numViews = len(G)
    numSamples = G[0].shape[1]

    out = [0] * numSamples
    for i in range(numViews):
      for j in range(i + 1, numViews):
        out = out + np.sum(G[i] * G[j], axis=0) - gamma * np.linalg.norm(S[i], axis=0) * np.linalg.norm(S[j], axis=0)

    return out

  def detector(self, FeaturesList, params ): #k=3, beta=0.6, gamma=0.1, maxIter=200):
    '''
    Implements Consensus Regularized Multi-view Outlier Detection algorithm

    Parameters: FeatureList : list
                    Each element of the list is a matrix N x D, where N is the number of samples and D is de dimension of feature vectors
    '''

    numSamples = len(FeaturesList[0])
    k, beta, gamma = self.getParams(params)
    numViews = len(FeaturesList)
    X = []
    G = []
    M = []
    H = []
    S = []
    Y = []
    GlobalFeatureMatrix = np.zeros((numSamples, 0))
    for taula in FeaturesList:
      trainIds = []
      featureMatrix = []
      for x in taula:
        trainIds.append(int(x.id))
        featureMatrix.append(x.features)

    featureMatrix = np.array(featureMatrix)
    X.append(np.transpose(featureMatrix))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(featureMatrix)
    j = kmeans.predict(featureMatrix)
    g = np.zeros((k, numSamples))
    g[j, range(numSamples)] = 1

    G.append(g)
    M.append(np.eye(k))
    H.append(np.zeros((featureMatrix.shape[1], k)))
    S.append(np.zeros((featureMatrix.shape[1], numSamples)))
    Y.append(np.zeros((featureMatrix.shape[1], numSamples)))

    GlobalFeatureMatrix = np.hstack((GlobalFeatureMatrix, featureMatrix))

    kmeans = KMeans(n_clusters=k, random_state=0).fit(GlobalFeatureMatrix)
    j = kmeans.predict(GlobalFeatureMatrix)
    GS = np.zeros((k, numSamples))
    GS[j, range(numSamples)] = 1

    i = 0
    E = 1000
    while E > self.eps and i < self.maxIter:
      i = i + 1
      # X = []

      for v, (x, h, g, s, y) in enumerate(zip(X, H, G, S, Y)):
        # x = np.matmul(h,g) + s
        # X.append(x)
        # Update S
        S[v] = self.updateS(x, h, g, y, self.mu)

      # Update G
      G = self.updateG(X, H, S, M, Y, GS, mu=self.mu, beta=self.beta)

      # Update G*
      GS = self.updateGS(M, G, beta=self.beta)

      # Update M
      M = self.updateM(G, GS)

      # Update H
      for v, (x, g, s, y) in enumerate(zip(X, G, S, Y)):
        H[v] = self.updateH(x, g, s, y, self.mu)

      # Update Y
      E = 0
      for v, (x, g, h, s, y) in enumerate(zip(X, G, H, S, Y)):
        Y[v] = y + self.mu * (x - np.matmul(h, g) - s)
        E = max(E, np.linalg.norm(x - np.matmul(h, g) - s, ord=np.Inf))

      # update mi
      self.mu = min(self.rho * self.mu, self.mu_max)

    # print i,E

    # Outlier measurement
    out = np.array(self.score(G, S, gamma=gamma))

    return out
