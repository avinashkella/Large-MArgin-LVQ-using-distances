# Large-MArgin-LVQ-using-distances
In large margin LVQ the training is based on prototypes and In this we predict data based on margin.

----------------------------------------------------------------------------------------------------------------------------------
Learning Vector Quantization(LVQ) is well-known for Supervised Vector Quantization. Large margin LVQ is to maximize the distance of sample margin or to maximize the distance between decision hyperplane and datapoints.

## Pseudo-code

1) Get data with labels.
2) Initialize prototypes with labels.
3) Calculate Euclidean distance between datapoints and prototypes.
4) Calculate closest correct matching prototype for every data point and also calculate <img src="http://latex.codecogs.com/svg.latex?|P_k|" title="http://latex.codecogs.com/svg.latex?|P_k|" /> is the number of data points for which <img src="http://latex.codecogs.com/svg.latex?w_k" title="http://latex.codecogs.com/svg.latex?w_k" /> is the closest prototype with same label.
5) Calculate <img src="http://latex.codecogs.com/svg.latex?1_{P_k}" title="http://latex.codecogs.com/svg.latex?1_{P_k}" />, A vector which has **1** where data point has closest prototype otherwise zero.
6) Compute <img src="http://latex.codecogs.com/svg.latex?A_k" title="http://latex.codecogs.com/svg.latex?A_k" />

*  The index <img src="http://latex.codecogs.com/svg.latex?A_k[i,&space;K*i&plus;l]" title="http://latex.codecogs.com/svg.latex?A_k[i, K*i+l]" />, should be **+1** if data point *i* is in <img src="http://latex.codecogs.com/svg.latex?|P_k|" title="http://latex.codecogs.com/svg.latex?|P_k|" />, i.e. if prototype *k* is the closest prototype to data point *i* with the same label, and if prototype *l* has a different label.

* The index <img src="http://latex.codecogs.com/svg.latex?A_k[i,&space;K*i&plus;k]" title="http://latex.codecogs.com/svg.latex?A_k[i, K*i+k]" /> should be **-1** if datapoint *i* has a different label than prototype *k*.

* The index <img src="http://latex.codecogs.com/svg.latex?A_k[i,&space;K*i&plus;l]" title="http://latex.codecogs.com/svg.latex?A_k[i, K*i+l]" /> should be zero in all other cases. So most of <img src="http://latex.codecogs.com/svg.latex?A_k" title="http://latex.codecogs.com/svg.latex?A_k" /> is zero.

7) Compute the cost function:

<img src="http://latex.codecogs.com/svg.latex?E&space;=&space;\min_{\vec{\lambda}&space;\in&space;\mathbb{R}^{m\cdot&space;K}}&space;\hspace{2mm}&space;\frac{1}{2}&space;\vec{\lambda}^T&space;\cdot&space;\left(C&space;\cdot&space;I&space;-&space;\sum_{k=1}^{K}&space;\textbf{A}_{k}^T&space;\cdot&space;\frac{D}{|P_k|}&space;\cdot&space;\textbf{A}_{k}\right)\cdot&space;\vec{\lambda}&space;-&space;\left(\gamma&space;\cdot&space;\vec{1}^T&space;&plus;&space;\sum_{k=1}^{K}&space;\vec{1}_{P_k}^T&space;\cdot&space;\frac{D}{|P_k|}&space;\cdot&space;\textbf{A}_{k}\right)&space;\cdot&space;\vec{\lambda}" title="http://latex.codecogs.com/svg.latex?E = \min_{\vec{\lambda} \in \mathbb{R}^{m\cdot K}} \hspace{2mm} \frac{1}{2} \vec{\lambda}^T \cdot \left(C \cdot I - \sum_{k=1}^{K} \textbf{A}_{k}^T \cdot \frac{D}{|P_k|} \cdot \textbf{A}_{k}\right)\cdot \vec{\lambda} - \left(\gamma \cdot \vec{1}^T + \sum_{k=1}^{K} \vec{1}_{P_k}^T \cdot \frac{D}{|P_k|} \cdot \textbf{A}_{k}\right) \cdot \vec{\lambda}" />

<img src="http://latex.codecogs.com/svg.latex?such&space;that&space;:math:`\vec{\lambda}&space;\geq&space;0`&space;and&space;&space;:math:`\vec{1}^T&space;\cdot&space;\textbf{A}_{k}^T&space;\cdot&space;\vec{\lambda}&space;=&space;0`,&space;:math:`\forall&space;k&space;\in&space;\{1,\dots,K\}`" title="http://latex.codecogs.com/svg.latex?such that :math:`\vec{\lambda} \geq 0` and :math:`\vec{1}^T \cdot \textbf{A}_{k}^T \cdot \vec{\lambda} = 0`, :math:`\forall k \in \{1,\dots,K\}`" />

8) Finally updates the prototypes:
<p align="center">
  <img src="http://latex.codecogs.com/svg.latex?\lambda(t&plus;1)&space;=&space;\lambda(t)&space;-&space;\eta&space;\frac{\partial&space;E}{\partial&space;w(t)}" title="http://latex.codecogs.com/svg.latex?\lambda(t+1) = \lambda(t) - \eta \frac{\partial E}{\partial w(t)}" />
</p>

## Installation
1) Clone this repository.
2) Make sure that `numpy`, `matplotlib`, `Scikit Learn` should be installed.
3) Go to folder and run the **python3 lmlvq_call.py** command.
