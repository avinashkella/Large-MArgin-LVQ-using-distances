#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 09:46:20 2020

@author: avinash
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import multi_dot

class LMLVQ:
    """
    Large margin LVQ is to maximize the distance of sample margin or
    to maximize the distance between decision hyperplane and data point.

    Attributes
    ----------
    prototype_per_class:
        The number of prototypes per class to be learned.

    """

    def __init__(self, prototype_per_class=1):
        """Init LMLVQ with prototypes per class is 1."""
        self.prototype_per_class = prototype_per_class

    prt_labels = np.array([])
    update_beta = np.array([])

    def normalization(self, input_data):
        """
        Normalize the data between range 0 and 1.

        Args:
        ----
        input_value:
            A n x m matrix of input data.

        Return:
        ------
        normalized_data:
            A n x m matrix with values between 0 and 1.

        """
        minimum = np.amin(input_data, axis=0)
        maximum = np.amax(input_data, axis=0)
        normalized_data = (input_data - minimum)/(maximum - minimum)
        return normalized_data

    def prt(self, input_data, data_labels, prototype_per_class):
        """
        Calculate prototypes with labels either at mean or randomly depends
        on prototypes per class.

        Args:
        ----
        input_value:
            A n x m matrix of datapoints.
        data_labels:
            A n-dimensional vector containing the labels for each
            datapoint.
        prototypes per class:
            The number of prototypes per class to be learned. If it is
            equal to 1 then prototypes assigned at mean position
            else it assigns randolmy.

        Return:
        ------
        prototype_labels:
            A n-dimensional vector containing the labels for each
            prototype.
        prototypes:
            A n x m matrix of prototyes.for training.
        lambda:
            A n*m array of zeros

        """
        #calculate prototype_labels
        prototype_labels = np.unique(data_labels)
        prototype_labels = list(prototype_labels) * prototype_per_class
        #calculate prototypes
        prt_labels = np.expand_dims(prototype_labels, axis=1)
        expand_dimension = np.expand_dims(np.equal(prt_labels, data_labels),
                                          axis=2)
        count = np.count_nonzero(expand_dimension, axis=1)
        proto = np.where(expand_dimension, input_data, 0)
        #if prototype_per_class is 1 then assign it to mean else assign prototypes randomly
        prototypes_array = []
        if prototype_per_class == 1:
            prototypes = np.sum(proto, axis=1)/count
        else:
            for lbl in range(len(prototype_labels)):
                x_values = input_data[data_labels == prototype_labels[lbl]]
                num = np.random.choice(x_values.shape[0], 1, replace=False)
                prototypes = x_values[num, :]
                prototypes_array.append(prototypes)
            prototypes = np.squeeze(np.array(prototypes_array))
        #save prototype labels
        self.prt_labels = prototype_labels
        lam = np.zeros(len(input_data)*len(prototype_labels))
        return self.prt_labels, prototypes, lam

    def euclidean_dist(self, input_data, prototypes):
        """
        Calculate squared Euclidean distance between datapoints and
        prototypes.

        Args:
        ----
        input_data:
            A n x m matrix of datapoints.
        prototpes:
            A n x m matrix of prototyes of each class.

        Return:
        ------
        eu_dist:
            A n x m matrix with Euclidean distance between datapoints and
            prototypes.
        D:
            A n x n matrix with Euclidean distance between datapoints.

        """
        expand_dimension = np.expand_dims(input_data, axis=1)
        distance = expand_dimension - prototypes
        distance_square = np.square(distance)
        sum_distance = np.sum(distance_square, axis=2)
        eu_dist = np.sqrt(sum_distance)
        D = euclidean_distances(input_data, input_data)
        return eu_dist, D

    def mod_Pk(self, input_data, data_labels, prototype_labels, eu_dist):
        """
        Calculate the number of datapoints for which w_k is closest prototype.

        Parameters
        ----------
        input_data:
            A n x m matrix of datapoints.
        data_labels:
            A n-dimensional vector containing the labels for each
            datapoint.
        prototype_labels:
            A n-dimensional vector containing the labels for each
            prototype.
        eu_dist:
            A n x m matrix with Euclidean distance between datapoints and
            prototypes.

        Returns
        -------
        w_plus_index:
            A n-dimensional vector containing the indices for nearest
            prototypes to datapoints with same label.
        pk:
            A list  of numbers of datapoints for which w_k is closest
            prototype.

        """
        expand_dimension = np.expand_dims(prototype_labels, axis=1)
        label_transpose = np.transpose(np.equal(expand_dimension, data_labels))

        # distance of matching prototypes
        plus_dist = np.where(label_transpose, eu_dist, np.inf)

        # index of minimum distance for best matching prototypes
        w_plus_index = np.argmin(plus_dist, axis=1)

        #calculate |P_k|
        pk = []
        for i in range(len(prototype_labels)):
            pull_values = input_data[w_plus_index == i]
            pk.append(len(pull_values))
        pk = np.array(pk)
        return w_plus_index, pk

    def one_p_k(self, input_data, prototype_labels, w_plus_index):
        """
        A vector which has 1 where data point has closest prototype otherwise
        zero.

        Parameters
        ----------
        input_data:
            A n x m matrix of datapoints.
        prototype_labels:
            A n-dimensional vector containing the labels for each
            prototype.
        w_plus_index:
            A n-dimensional vector containing the indices for nearest
            prototypes to datapoints with same label.

        Returns
        -------
        pk:
            A m-dimensional vector which has 1 for data point has
            closest prototype otherwise zero.

        """
        #initialize 1_{P_k}
        pk = np.zeros((len(input_data), len(prototype_labels)))
        for i in range(np.shape(pk)[0]):
            #1 when i \in P_k and else 0
            value = w_plus_index[i]
            pk[i][value] = 1
        return pk


    def A_k(self, input_data, prototype_labels, w_plus_index):
        """
        A_K used to translating the lambda numbers to the beta numbers.

        The index A_k[i, K*i+l] should be +1 if data point i is in P_k, i.e.
        if prototype k is the closest prototype to data point i with the same
        label, _and_ if prototype l has a different label.

        The index A_k[i, K*i+k] should be -1 if datapoint i has a different
        label than prototype k.

        The index A_k[i, K*i+l] should be zero in all other
        cases. So most of A_k is zero.

        Parameters
        ----------
        input_data:
            A n x m matrix of datapoints.
        prototype_labels:
            A n-dimensional vector containing the labels for each
            prototype.
        w_plus_index:
            A n-dimensional vector containing the indices for nearest
            prototypes to datapoints with same label.

        Returns
        -------
        a_k:
            A N x (N*K) dimentional array where N is data points and K are
            prototypes. This is for every prototype

        """
        a_k = []
        for k in range(len(prototype_labels)):
            value = prototype_labels[k]
            #initialize A_k
            ak = np.zeros((len(input_data), len(input_data) *
                           len(prototype_labels)))
            for i in range(len(input_data)):
                #calculate i in P_k
                if w_plus_index[i] == k:
                    for l in range(len(prototype_labels)):
                        x_values = prototype_labels[k]
                        y_values = prototype_labels[l]
                        if x_values != y_values:
                            ak[i, len(prototype_labels)*i+l] = 1
                #calculate i in N_k
                else:
                    ak[i, len(prototype_labels)*i+k] = -1
            a_k.append(ak)
        return a_k

    def cost_function(self, lam, C, gamma, ak, D, one_pk, pk,
                      prototype_labels, kappa):
        """
        Calculate cost function of LMLVQ.

        cost function: 0.5 * \lambda^T * H * \lambda - q^T * \lambda
        where H = C*I - \sum_k (A_k * (D/P_k) * A_k)
        and q = gamma * 1.T + \sum_k (1_{P_k} * (D/P_k) * A_k)

        Parameters
        ----------
        lam:
            A N*K lambda vector where N is the number of data points and K is
            the number of prototypes
        C:
            The regularization constant.
        gamma:
            The margin parameter.
        ak:
            A N x (N*K) dimentional array where N is data points and K are
            prototypes. This is for every prototype.
        D:
            A n x n matrix with Euclidean distance between datapoints.
        one_pk:
            A m-dimensional vector which has 1 for data point has
            closest prototype otherwise zero.
        pk:
            A list  of numbers of datapoints for which w_k is closest
            prototype.
        prototype_labels:
            A n-dimensional vector containing the labels for each
            prototype.
        kappa:
            A hyperparameter.

        Returns
        -------
        H:
             A n x n matrix of result C*I - \sum_k (A_k * (D/P_k) * A_k)
        q:
             A n-dimensional vector of result:
             gamma * 1.T + \sum_k (1_{P_k} * (D/P_k) * A_k)
        cf:
            A n x n matrix of result of cost function.

        """

        #second condtion for cost function: 1.T * A[k] * lambda = 0
        Ahat = []
        for k in range(len(prototype_labels)):
            x = np.sum(ak[k], 0)
            Ahat.append(x)
        ahat_dot = np.dot(np.transpose(Ahat), Ahat)
        result = kappa * ahat_dot

        #calculate H
        I = np.eye(len(lam))
        sum_H = []
        for k in range(len(prototype_labels)):
            divide = D/pk[k]
            A = np.transpose(ak[k])
            H = multi_dot([A, divide, ak[k]])
            sum_H.append(H)
        sum_H = np.sum(sum_H, 0)
        H = C*I - sum_H
        H = np.add(H, result)

        #calculate q
        one_transpose = np.ones(len(lam))
        sum_q = []
        for k in range(len(prototype_labels)):
            divide = D/pk[k]
            A = np.transpose(one_pk[:, k])
            q = multi_dot([A, divide, ak[k]])
            sum_q.append(q)
        sum_q = np.sum(sum_q, 0)
        q = gamma * one_transpose + sum_q

        #calculate 0.5 * \lambda^T * H * \lambda - q^T * \lambda
        cf = (0.5 * np.transpose(lam) * H * lam) - (q * lam)

        return H, q, cf

    def beta_k(self, lam, prototype_labels, ak, one_pk):
        """
        Use to compute Euclidean distance between data points and prototypes.

        beta_k = A_k * lambda + 1_{P_k}

        Parameters
        ----------
        lam:
            A N*K lambda vector where N is the number of data points and K is
            the number of prototypes
        prototype_labels:
            A n-dimensional vector containing the labels for each
        ak:
            A N x (N*K) dimentional array where N is data points and K are
            prototypes. This is for every prototype.
        one_pk:
            A m-dimensional vector which has 1 for data point has
            closest prototype otherwise zero.

        Returns
        -------
        result:
            A K x (N x 1) matrix with K is number of prototypes, N is number
            of datapoints.

        """
        result = []
        for k in range(len(prototype_labels)):
            beta = np.dot(ak[k], lam) + one_pk[:, k]
            result.append(beta)
        return result

    def update(self, lam, H, q, learning_rate):
        """
        To update the lambda vector.

        gradient = H * \lambda - q
        lambda(t+1) = lambda - learning rate * gradient

        Parameters
        ----------
        lam:
            A N*K lambda vector where N is the number of data points and K is
            the number of prototypes
        H:
             A n x n matrix of result C*I - \sum_k (A_k * (D/P_k) * A_k)
        q:
             A n-dimensional vector of result:
             gamma * 1.T + \sum_k (1_{P_k} * (D/P_k) * A_k)
        learning_rate:
                The step size.

        Returns
        -------
        lam_update:
            A N*K updated lambda vector where N is the number of data points
            and K is the number of prototypes.

        """
        #calculate gradient = H * \lambda - q
        value = np.dot(H, lam) - q
        #update \lambda using stochastic gradient
        lam_update = lam - learning_rate * value
        return lam_update

    def d_i_k(self, input_data, prototype_labels, D, beta):
        """
        Calculate distance between data points and prototypes.

        Formula:
            d_{i,k} = \sum_j \beta_k[j] * D[i, j] / np.sum(beta_k) - 0.5 *
                     np.dot(beta_k, np.dot(D, beta_k)) / (np.sum(beta_k) ** 2)

        Parameters
        ----------
        input_data:
            A n x m matrix of datapoints.
        prototype_labels:
            A n-dimensional vector containing the labels for each
        D:
            A n x n matrix with Euclidean distance between datapoints.
        beta:
            A K x (N x 1) matrix with K is number of prototypes, N is number
            of datapoints.

        Returns
        -------
        dik:
            A n x m matrix with distance between datapoints and
            prototypes.

        """
        dik = np.zeros((len(input_data), len(prototype_labels)))
        #calculate left part numerator
        #\sum_j \beta_k[j] * D[i, j]
        result = []
        for k in range(len(prototype_labels)):
            final = []
            for i in range(len(input_data)):
                total = []
                for j in range(len(input_data)):
                    res = beta[k][j] * D[i, j]
                    total.append(res)
                final.append(np.sum(total, 0))
            result.append(final)

        #calculate left part denominator
        #\sum_j \beta_k[j]
        dividor_left = np.sum(beta, 1)
        #first part: \sum_j \beta_k[j] * D[i, j] / \sum_j \beta_k[j]
        value1 = []
        for i in range(len(dividor_left)):
            left = result[i] / dividor_left[i]
            value1.append(left)

        #calculate right side numerator
        num = []
        for k in range(len(prototype_labels)):
            A = np.transpose(beta[k])
            res = multi_dot([A, D, beta[k]])
            num.append(res)

        #calculate right part denominator
        dividor_right = np.square(np.sum(beta, 1))
        #second part: (\beta_k * D * \beta_k) / (sum_j \beta_k[j])**2
        value2 = num / dividor_right

        #calculate total = first_part - second_part
        eu_dist = []
        for k in range(len(value1)):
            total = value1[k] - 0.5 * (value2[k])
            eu_dist.append(total)

        #calculate d_ik
        for k in range(len(prototype_labels)):
            for i in range(len(input_data)):
                dik[i][k] = eu_dist[k][i]
        return dik

    def fit(self, input_data, data_labels, learning_rate, epochs, margin,
            constant, kappa):
        """
        Train the Algorithm.

        Parameters
        ----------
        input_data:
            A n x m matrix of datapoints.
        data_labels:
            A n-dimensional vector containing the labels for each
            datapoint.
        learning_rate:
                The step size.
        epochs:
                The maximum number of optimization iterations.
        margin:
            The margin parameter.
        Constant:
            The regularization constant.
        kappa:
            A hyperparameter.

        Returns
        -------
        beta:
            A K x (N x 1) updated beta matrix with K is number of prototypes,
            N is number of datapoints.

        """
        #normalize the training data
        normalized_data = self.normalization(input_data)
        #initialize lam and get prototypes with labels
        prototype_labels, prototypes, lam = self.prt(normalized_data,
                                                     data_labels,
                                                     self.prototype_per_class)
        #initilize euclidean distance between data point and prototypes
        eu_dist, D = self.euclidean_dist(normalized_data, prototypes)

        #iteration to train model
        for i in range(epochs):
            #calculate |P_k| and closest prototype
            w_plus_index, pk = self.mod_Pk(normalized_data, data_labels,
                                           prototype_labels, eu_dist)
            #calculate 1_{P_k}
            one_pk = self.one_p_k(normalized_data, prototype_labels,
                                  w_plus_index)
            #calculate A_k matrix
            ak = self.A_k(normalized_data, prototype_labels, w_plus_index)

            #cost function
            H, q, cf = self.cost_function(lam, constant, margin, ak, D,
                                          one_pk, pk, prototype_labels, kappa)
            #calculate beta
            beta = self.beta_k(lam, prototype_labels, ak, one_pk)

            #first condition of lambda>=0
            for k in range(len(lam)):
                if lam[k] < 0:
                    lam[k] = 0

            #update lam to update beta
            lam = self.update(lam, H, q, learning_rate)

            #use dik as a distance between data point and prototypes
            eu_dist = self.d_i_k(normalized_data, prototype_labels, D, beta)

        #save beta in order to test data
        self.update_beta = beta
        #print(self.update_beta)
        return beta

    def predict(self, input_value, input_data):
        """
        Predicts the labels for the data represented by the
        given test-to-training distance matrix.

        Parameters
        ----------
        input_value:
            A n x m matrix of distances from the test to the training
            datapoints.
        input_data:
            A n x m matrix of datapoints.

        Returns
        -------
        ylabel:
            A n-dimensional vector containing the predicted labels for each
            datapoint.

        """
        #input_value is a testing data
        #input_data is a training data

        #create empty array
        final = np.zeros((len(input_value), len(self.prt_labels)))

        #normalize the training and testing data
        input_value = self.normalization(input_value)
        input_data = self.normalization(input_data)
        #get the updated beta
        beta = self.update_beta

        #calculate d[i] = d(x, x_i)
        d, dist = self.euclidean_dist(input_value, input_data)
        #D is the distance matrix between training data points
        D = euclidean_distances(input_data, input_data)

        #calculate sum of beta
        beta_sum = np.sum(beta, 1)

        # calculate z[k] = -0.5 * np.dot(beta[k], np.dot(D, beta[k])).
        z_k = []
        for k in range(len(beta)):
              z = -0.5 * multi_dot([beta[k], D, beta[k]])
              res = z/(beta_sum[k]**2)
              z_k.append(res)

        #calculate d(x, w_k) = np.dot(d, beta[k]) + z[k]
        result = []
        for k in range(len(beta)):
            #value = np.dot(d, beta[k]) / beta_sum[k]
            #res = value + z_k[k]
            res = np.dot(d, beta[k])/beta_sum[k] + z_k[k]
            result.append(res)

        #calculate final array
        for k in range(len(self.prt_labels)):
            for i in range(len(input_value)):
                final[i][k] = result[k][i]

        #use d_ik to calculate distance of test data
        #final =  self.d_i_k(input_value, self.prt_labels, D, beta)

        # #calculate the minimum distance
        z = np.argmin(final, axis=1)
        ylabel = []
        for k in range(len(z)):
            x = z[k]
            ylabel.append(self.prt_labels[x])
        return ylabel
