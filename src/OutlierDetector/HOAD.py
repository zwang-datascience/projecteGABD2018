# -*- coding: utf-8 -*-

import OutlierDetector as od

from scipy.sparse import csgraph
from scipy.sparse.linalg import svds
#from sklearn.cluster import KMeans
from scipy.spatial import distance as dist
#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

class HOAD(od.OutlierDetector):
    u'''

    This class implements the HOAD method that was introduced in [1]


    [1]	J. Gao, N. Du, W. Fan, D. Turaga, S. Parthasarathy, J. Han, "A multi-graph spectral framework for mining multi-source anomalies" in in Graph Embedding for Pattern Analysis, New York, NY, USA:Springer, pp. 205-228, 2013.

    Created on 1/9/2018

    @author: Oriol Ramos Terrades (oriolrt@cvc.uab.cat)
    @Institution: Computer Vision Center - Universitat Autonoma de Barcelona
    '''


    def __init__(self,numViews=2):
        '''
        Constructor
        '''
        self.sigma = 1
        self.similarity='euclidean'
        self.maxSamples = 2000
        super(HOAD, self).__init__(numViews)


    def computeFullyConnected(self,taula, similarity='gaussian', sigma=1):

        # taula.show(10)
        numSamples = len(taula)
        numFeatures = len(taula[0].features)
        trainIds = [0] * numSamples
        featureMatrix = np.zeros((numSamples, numFeatures))
        for i, x in enumerate(taula):
            trainIds[i] = int(x.id)
            if len(x.features) != numFeatures:
                print "eps!"
            featureMatrix[i, :] = x.features

        featureMatrix = np.array(featureMatrix)
        trainIds = range(len(trainIds))

        if similarity.lower() == 'euclidean':
            D = dist.squareform(dist.pdist(featureMatrix))
            meanDist = np.mean(D)
            varDist = np.var(D)
        elif similarity.lower() == 'gaussian':
            D = dist.squareform(dist.pdist(featureMatrix))
            meanDist = np.mean(D)
            # sigma = 1 #snp.var(D)
            # D = np.exp(-D**2/2/np.std(D))
            idx = np.argsort(np.exp(-D ** 2 / 2 / sigma))
        elif similarity.lower() == 'jaccard':
            D = dist.squareform(dist.pdist(featureMatrix, metric=similarity.lower()))

        A = []
        # Compute d'ajacency matrix for the training set
        for i in trainIds:
            for j in trainIds[i + 1:]:
                A = A + [
                    (int(i), int(j), {'edist': float(D[i, j]), 'gdist': np.exp(-(float(D[i, j])) ** 2 / 2 / sigma)})]
                # A = A +  [ (int(i),int(idx[i,pos][x]),{'edist': float(D[i,idx[i,pos]][x]), 'gdist': np.exp(-(float(D[i,idx[i,pos]][x])-meanDist)**2/2/varDist  )}) for x in range(0,len(pos))  ]

        # Build the mutual-KNN graph on the training dataset
        G = nx.Graph()
        G.add_nodes_from(trainIds)
        G.add_edges_from(A)

        nconn = nx.number_connected_components(G)

        return G

    def computeKnn(self,taula, k=40, similarity='gaussian', sigma=1):
        '''

        :param taula:
        :param k:
        :param similarity:
        :param sigma:
        :return:
        '''

        # taula.show(10)
        numSamples = len(taula)
        if k > numSamples / 2: print(
                "Number of samples (%d) lower than number of mutual friends (%d). Updating k \n" % (numSamples / 2, k))
        k = min(numSamples / 2 - 1, k)
        trainIds = []
        featureMatrix = []
        for x in taula:
            trainIds.append(int(x.id))
            featureMatrix.append(np.array(x.features))

        trainIds = range(len(trainIds))
        # ===========================================================================
        # trainIds = [ int(x.id) for x in taula]
        # featureMatrix = np.array( [ np.array(x.features) for x in taula] )
        # ===========================================================================

        if similarity.lower() == 'euclidean':
            D = dist.squareform(dist.pdist(featureMatrix))
            meanDist = np.mean(D)
            varDist = np.var(D)
            idx = np.argsort(D)[:, 1:k + 1]
        elif similarity.lower() == 'gaussian':
            D = dist.squareform(dist.pdist(featureMatrix))
            meanDist = np.mean(D)
            varDist = 1  # snp.var(D)
            # D = np.exp(-D**2/2/np.std(D))
            idx = np.argsort(np.exp(-D ** 2 / 2 / varDist))[:, -k - 1:-1]
        elif similarity.lower() == 'jaccard':
            D = dist.squareform(dist.pdist(featureMatrix, metric=similarity.lower()))
            idx = np.argsort(D)[:, -k - 1:-1]

        A = []

        # Compute d'ajacency matrix for the training set
        for i in range(0, len(trainIds)):
            pos = [x for x in range(0, k) if any((idx[idx[i, :]] == i)[x, :])]
            A = A + [(int(trainIds[i]), int(trainIds[idx[i, pos][x]]), {'edist': float(D[i, idx[i, pos]][x]),
                                                                        'gdist': np.exp(-(float(D[i, idx[i, pos]][
                                                                                                    x]) - meanDist) ** 2 / 2 / varDist)})
                     for x in range(0, len(pos))]
            # A = A +  [ (int(i),int(idx[i,pos][x]),{'edist': float(D[i,idx[i,pos]][x]), 'gdist': np.exp(-(float(D[i,idx[i,pos]][x])-meanDist)**2/2/varDist  )}) for x in range(0,len(pos))  ]

        # Build the mutual-KNN graph on the training dataset
        G = nx.Graph()
        G.add_nodes_from(trainIds)
        G.add_edges_from(A)

        nconn = nx.number_connected_components(G)

        return G

    def hoad(self, graphList, m=1, k=3, show=True):


        if len(graphList) == 1:
            A = sp(nx.to_numpy_matrix(graphList[0], range(max(graphList[0].nodes())), weight='gdist'))
            B = A
            K = int(k)  # + nx.number_connected_components(graphList[0])
            print "Computing anomaly detection with one graph layer. No anomalies should be detected"
        elif len(graphList) == 2:
            numSamples = max(max(graphList[0].nodes()), max(graphList[1].nodes())) + 1
            if numSamples > self.maxSamples:
                A = sp.coo_matrix(nx.to_numpy_matrix(graphList[0], range(numSamples), weight='gdist'))
                B = sp.coo_matrix(nx.to_numpy_matrix(graphList[1], range(numSamples), weight='gdist'))
                C = sp.eye(numSamples) * (np.mean(A[np.nonzero(A)]) + np.mean(B[np.nonzero(B)])) / 2
            else:
                A = nx.to_numpy_matrix(graphList[0], range(numSamples), weight='gdist')
                B = nx.to_numpy_matrix(graphList[1], range(numSamples), weight='gdist')
                C = np.eye(numSamples) * m
                # print "Computing anomaly detection with two graph layers. "

            K = int(k)  # + (nx.number_connected_components(graphList[0])+nx.number_connected_components(graphList[1]))/2
        else:
            print "Anomaly detection with more than 2 layers  is still not implemented"
            return 0

        if A.shape != B.shape:
            print "Adjacency matrices should have the same number of nodes"
            return 0

        # numSamples,_ = A.shape
        if numSamples > self.maxSamples:
            Z = sp.vstack((sp.hstack((A, C)), sp.hstack((C, B))))
        else:
            Z = np.vstack((np.hstack((A, C)), np.hstack((C, B))))

        # D=np.diag(np.sum(Z,1))
        if show:
            I, J, V = sp.find(Z)
            W = []
            for i, j, v in zip(I, J, V):
                W = W + [(i, j, {'dist': v})]

            G = nx.Graph()
            G.add_edges_from(W)

            pos = nx.spring_layout(G)  # positions for all nodes
            labels = {}
            for i, x in enumerate(range(0, numSamples) + range(0, numSamples)): labels[i] = r'$%d$' % (x)
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=range(0, numSamples),
                                   node_color='r',
                                   node_size=500,
                                   alpha=0.8)
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=range(numSamples, 2 * numSamples),
                                   node_color='b',
                                   node_size=500,
                                   alpha=0.8)
            nx.draw_networkx_labels(G, pos, labels, font_size=16)
            nx.draw_networkx_edges(G, pos, edgelist=W, alpha=0.5)
            plt.axis('off')
            # plt.savefig("labels_and_colors.png") # save as png
            plt.show()

        if numSamples > self.maxSamples:
            # L = D - Z
            # U,S,V = svds(sp.eye(numSamples*2) - Z)
            # k = numSamples
            U, S, _ = svds(csgraph.laplacian(Z), K, which='SM')

        else:
            U, S, V = np.linalg.svd(np.eye(numSamples * 2) - Z)

        u = U[:numSamples, -K:]
        v = U[numSamples:, -K:]

        # ===========================================================================
        # u = U[:numSamples,np.isnan(S) == False]
        # v = U[numSamples:,np.isnan(S) == False]
        # ===========================================================================

        # ===========================================================================
        # uu=np.matmul(A,u)
        # vv=np.matmul(B,v)
        #
        # ss = 1-np.sum(uu*vv,1)/np.sqrt(np.sum(uu*uu,1))/np.sqrt(np.sum(vv*vv,1))
        # ===========================================================================

        s = 1 - np.abs(np.sum(np.multiply(u, v), 1) / np.sqrt(np.sum(np.multiply(u, u), 1)) / np.sqrt(
            np.sum(np.multiply(v, v), 1)))

        # ===========================================================================
        # s = 1-u*v/np.abs(u)/np.abs(v)
        # ===========================================================================

        return s  # np.where(s<.5)[0].tolist()

    def detector(self, FeaturesList, params ):

        G = []
        for i, layer in enumerate(self.layers):
            if layer == 'Label': self.similarity = 'jaccard'
            if layer == 'Sigmoid': self.similarity = 'gaussian'
            # G.append(computeKnn(newFeatures[i], similarity=similarity, sigma=sigma))
            G.append(self.computeFullyConnected( FeaturesList[i], similarity=self.similarity, sigma=params['sigma']))

        # if loadFeatures: loadDescriptors(keyspace,table,featuresFile,hosts)
        return self.hoad(G, m=params['m'], k=params['k'], show=False)





