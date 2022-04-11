# Sequence and Video Prediction with Convolutional LSTMs

Junior year independent work project at Princeton
by Matthew Coleman

## Introduction

While humans cannot perfectly predict the future, they are able to predict very
near events at a high level, and this knowledge greatly aids them in planning
out their own actions, such as movements to take to reach a goal. On the one
hand, some situations are very predictable, such as the motion of a pendulum
swinging in a clock, or the spinning of a merry-go-round. These are mostly
deterministic events, that is, they have a high likelihood of continuing in a
predictable way. Other situations are more unpredictable, such as the moves and
strikes of an opponent boxer, or the paths of people walking on a busy street.
These events are mostly stochastic, meaning that they have only a lowlikelihood
of proceeding in any given way, and the longer the sequence goes on, the more
unlikely it will be that a person--or a machine learning model--can predict
what will happen next.

The task of video prediction in computer vision is a self-supervising task that
involves splitting a sequence of video frames into an input data and label
pair. The task is self-supervising because the first half of the data is used
as the input and the second half is used as the label, that is, the model
directly computes a loss function between it's own output and the actual ground
truth frames from the original sequence. In effect, the model learns solely
from real-world data, without any human intervention in the data-labeling process.

## Architectures

There are a wide range of model architectures which are suited to
the task of video-prediction, however they can be grouped by several key
features.

### Stochastic Models

### Discriminative Models

Discriminative models are the most basic computer vision and machine learning
models. An example could be a fully-connected linear model that classifies
MNIST characters as their corresponding digits (0-9), by taking input data and
backpropagating with a loss function computed between the correct label and the
model's own inference. 

In doing this, discriminative models make the assumption that the true labels y
are related to the input data x by a conditional probability distribution
p(y|x), which is a perfectly fine assumption to make in most cases. If the input data
consists of a handwritten digit, then it is reasonable to conclude that there
would exists some mapping from handwritten digits to the number label of those
digits.

### Generative Models

Generative models are different from discriminative models in many ways, and in
fact, many generative model architectures actually incorporate some kind of
discriminative model that performs a key function in its inference and
training. Instead of directly determining the probability distribution p(y|x),
generative models make the assumption that the input data are conditionally
related to the output data, that is, they attempt to determine the probability
distribution p(x|y).

## Experiments

### Sequence Prediction

The most fundamental capability of a prediction model is the ability to predict
simple, one-dimensional, deterministic data. Humans can do this task quite well
simply by keeping rhythm while tapping their feet to a song, for example, and
the ability to predict oncoming beats is something that allows them to stay on
rhythm. This task could be proposed to a machine learning model by giving it a
simple array of numbers where time without a beat is represented by a zero and
time with a beat is represented by a 1, and the 1s are somehow predictably
spaced from each other.

## Results

## Conclusion

## References
