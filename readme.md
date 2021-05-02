## Neural network with error backpropagation - C implementation

A piece of code that creates a NN, trains it and tests it in a set of examples by Zalando Research
as found on [kaggle.com](https://www.kaggle.com/zalando-research/fashionmnist/data)

#### Accuracy

The accuracy with 60,000 images on 10,000 test images is 53%, rising to 76% when trained multiple (10) times
on the same set.

#### Compile

`gcc nn_bp.c -o nn_bp -lm -O3`

###### Todo:

- Make the number of hidden layers flexible
- Include multiple activation functions to choose from
