# Large-MArgin-LVQ-using-distances
In large margin LVQ the training is based on prototypes and In this we predict data based on margin.

----------------------------------------------------------------------------------------------------------------------------------
Learning Vector Quantization(LVQ) is well-known for Supervised Vector Quantization. Large margin LVQ is to maximize the distance of sample margin or to maximize the distance between decision hyperplane and datapoints.

## Pseudo-code

1) Get data with labels.
2) Initialize prototypes with labels.
3) Calculate Euclidean distance between datapoints and prototypes.
4) Calculate closest correct matching prototype for every data point and also calculate <img src="http://latex.codecogs.com/svg.latex?|P_k|" title="http://latex.codecogs.com/svg.latex?|P_k|" /> is the number of data points for which :math:`w_k` is the closest prototype with same label.
5) Extract distances between datapoints and prototypes with different label by using Euclidean distance.
6) Compute the cost function:

<p align="center">
  <img src="http://latex.codecogs.com/svg.latex?E&space;=&space;\overbrace{\sum_{i=1}^{m}&space;d_{i}^&plus;}^{pull}&space;&plus;&space;\frac{1}{2C}&space;\cdot&space;\overbrace{\sum_{i=1}^{m}&space;\sum_{l&space;\in&space;I_i}&space;ReLU(d_{i}^&plus;&space;-&space;d_{i,l}&space;&plus;&space;\gamma)^2}^{push}&space;" title="http://latex.codecogs.com/svg.latex?E = \overbrace{\sum_{i=1}^{m} d_{i}^+}^{pull} + \frac{1}{2C} \cdot \overbrace{\sum_{i=1}^{m} \sum_{l \in I_i} ReLU(d_{i}^+ - d_{i,l} + \gamma)^2}^{push} " />
</p>

7) Finally updates the prototypes:
<p align="center">
  <img src="http://latex.codecogs.com/svg.latex?w(t&plus;1)&space;=&space;w(t)&space;-&space;\eta&space;\frac{\partial&space;E}{\partial&space;w(t)}" title="http://latex.codecogs.com/svg.latex?w(t+1) = w(t) - \eta \frac{\partial E}{\partial w(t)}" />
</p>

## Installation
1) Clone this repository.
2) Make sure that `numpy`, `matplotlib`, `Scikit Learn` should be installed.
3) Go to folder and run the **python3 lmlvq_call.py** command.
