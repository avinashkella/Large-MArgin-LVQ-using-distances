.. LMLVQ using distance documentation master file, created by
   sphinx-quickstart on Wed Oct 21 09:40:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LMLVQ using distance's documentation!
================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Algorithm Description
=====================

Learning Vector Quantization(LVQ) is well-known for Supervised Vector Quantization. Large margin LVQ is to maximize the distance of sample margin or to maximize the distance between decision hyperplane and datapoints.

***********
Pseudo-code
***********

1. Get data with labels e.g. :math:`x \in X, |X| = n, X \subset R^n` , where *X* are datapoints and :math:`c(X) \in C`, where *C* are data labels.
2. Initialize prototypes with labels e.g. :math:`w \in R^n`, where *w* are prototypes and :math:`c(w) \in C`, where *C* are prototype labels.
3. Calculate Euclidean distance between datapoints and prototypes.

.. math::
   	d_{i,j} = d_{E}(x_i , w_j) = \sqrt{\sum (x_i - w_k)^2}

4. Calculate closest correct matching prototype for every data point and also calculate :math:`|P_k|` is the number of data points for which :math:`w_k` is the closest prototype with same label.
5. Calculate :math:`1_{P_k}`, A vector which has **1** where data point has closest prototype otherwise zero.
6. Compute :math:`A_k`,

+ The index :math:`A_k[i, K*i+l]`, should be **+1** if data point *i* is in :math:`|P_k|`, i.e. if prototype *k* is the closest prototype to data point *i* with the same label, and if prototype *l* has a different label.

+ The index :math:`A_k[i, K*i+k]` should be **-1** if datapoint *i* has a different label than prototype *k*.

+ The index :math:`A_k[i, K*i+l]` should be zero in all other cases. So most of :math:`A_k` is zero.

7. Compute the cost function:

.. math::
   E = \min_{\vec{\lambda} \in \mathbb{R}^{m\cdot K}} \hspace{2mm} \frac{1}{2} \vec{\lambda}^T \cdot \left(C \cdot I - \sum_{k=1}^{K} \textbf{A}_{k}^T \cdot \frac{D}{|P_k|} \cdot \textbf{A}_{k}\right)\cdot \vec{\lambda} - \left(\gamma \cdot \vec{1}^T + \sum_{k=1}^{K} \vec{1}_{P_k}^T \cdot \frac{D}{|P_k|} \cdot \textbf{A}_{k}\right) \cdot \vec{\lambda}

+ such that :math:`\vec{\lambda} \geq 0` and  :math:`\vec{1}^T \cdot \textbf{A}_{k}^T \cdot \vec{\lambda} = 0`, :math:`\forall k \in \{1,\dots,K\}`

8. Finally updates the prototypes:

.. math::
	\lambda(t+1) = \lambda(t) - \eta \frac{\partial E}{\partial w(t)}
	


Installation Requirements
=========================

Following are the basic requirements to run this program:

1. `python <www.python.org>`_ with minimum version 3.8.
2. `numpy <https://numpy.org/install/>`_ with minimum version 1.19.0.
3. `matplotlib <https://matplotlib.org/users/installing.html>`_.
4. `Scikit Learn <https://scikit-learn.org/stable/>`_ .

**********
Execution
**********

* Open the **lmlvq_call.py** file and set parameters with margin, prototypes per class and epochs then run it.


Classes and Functions
=====================

.. automodule:: lmlvq_distance
    :members:
    :undoc-members:
    :show-inheritance:	

