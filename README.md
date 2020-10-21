

<img src="http://latex.codecogs.com/svg.latex?\frac{1}{2}&space;\vec{\lambda}^T&space;\cdot&space;\left(C&space;\cdot&space;I&space;-&space;\sum_{k=1}^{K}&space;\textbf{A}_{k}^T&space;\cdot&space;\frac{D}{|P_k|}&space;\cdot&space;\textbf{A}_{k}\right)\cdot&space;\vec{\lambda}&space;-&space;\left(\gamma&space;\cdot&space;\vec{1}^T&space;&plus;&space;\sum_{k=1}^{K}&space;\vec{1}_{P_k}^T&space;\cdot&space;\frac{D}{|P_k|}&space;\cdot&space;\textbf{A}_{k}\right)&space;\cdot&space;\vec{\lambda}&space;" title="http://latex.codecogs.com/svg.latex?\frac{1}{2} \vec{\lambda}^T \cdot \left(C \cdot I - \sum_{k=1}^{K} \textbf{A}_{k}^T \cdot \frac{D}{|P_k|} \cdot \textbf{A}_{k}\right)\cdot \vec{\lambda} - \left(\gamma \cdot \vec{1}^T + \sum_{k=1}^{K} \vec{1}_{P_k}^T \cdot \frac{D}{|P_k|} \cdot \textbf{A}_{k}\right) \cdot \vec{\lambda} " />

<img src="http://latex.codecogs.com/svg.latex?\hspace{2mm}&space;\vec{\lambda}&space;\geq&space;0&space;\hspace{2mm}&space;and&space;\hspace{2mm}&space;\vec{1}^T&space;\cdot&space;\textbf{A}_{k}^T&space;\cdot&space;\vec{\lambda}&space;=&space;0\hspace{1cm},&space;\forall&space;k&space;\in&space;\{1,\dots,K\}" title="http://latex.codecogs.com/svg.latex?\hspace{2mm} \vec{\lambda} \geq 0 \hspace{2mm} and \hspace{2mm} \vec{1}^T \cdot \textbf{A}_{k}^T \cdot \vec{\lambda} = 0\hspace{1cm}, \forall k \in \{1,\dots,K\}" />

8) Finally updates the prototypes:
<p align="center">
  <img src="http://latex.codecogs.com/svg.latex?\lambda(t&plus;1)&space;=&space;\lambda(t)&space;-&space;\eta&space;\frac{\partial&space;E}{\partial&space;w(t)}" title="http://latex.codecogs.com/svg.latex?\lambda(t+1) = \lambda(t) - \eta \frac{\partial E}{\partial \lambda(t)}" />
</p>

## Installation
1) Clone this repository.
2) Make sure that `numpy`, `matplotlib`, `Scikit Learn` should be installed.
3) Go to folder and run the **python3 lmlvq_call.py** command.
