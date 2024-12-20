![License](https://img.shields.io/badge/license-MIT-blue) ![Downloads](https://img.shields.io/pypi/dm/your-package-name)
# CREPE Pitch Tracker
Using Pytorch to reproduce the CREPE model, with the model structure based on the paper "CREPE: A Convolutional Representation for Pitch Estimation" by Jong Wook Kim, Justin Salamon, Peter Li, and Juan Pablo Bello.

# Train CREPE 
Due to certain doubts about the model structure in the [[paper](https://arxiv.org/abs/1802.06182)]?

A 6-layer convolutional network and a fully connected layer were used during actual training, with Dropout technique applied for regularization. ReLU activation functions were added in the convolutional layers, cross-entropy loss function was used, and a confidence normal distribution was employed to accelerate network fitting.

# References
[[1](https://arxiv.org/abs/1802.06182)] Jong Wook Kim, Justin Salamon, Peter Li, and Juan Pablo Bello. _CREPE: A Convolutional Representation for Pitch Estimation_. arXiv:1802.06182 [eess.AS] 17 Feb 2018

# License
This project is based on the structure of the CREPE project.
