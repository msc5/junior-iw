
# IW Notes

## Report Organization
- LSTM Architecture and Gates
- CONV-LSTM alterations and implementations
- SAVP ?

## To-Do 
- [FutureGAN](https://medium.com/analytics-vidhya/review-of-futuregan-predict-future-video-frames-using-generative-adversarial-networks-gans-3120d90d54e0)
- [PixelCNN](https://github.com/openai/pixel-cnn)

# Datasets
- Moving MNIST
- KTH
- Cityscapes?

# Meeting Notes

1. Currently have data on training and prediction from normal LSTM (text
   prediction) and Conv LSTM both my own implementation and reference
   implementations. Currently only data from training on MovingMNIST and KTH.
   * Want to include datasets made purely for video prediction (others are for
	 classification). I.e., Standard Bouncing Balls, Robotic Pushing Dataset,
	 BAIR

2. It's relatively easy to git clone other implentations and test them, so I
   think I will move forward with breadth rather than depth (e.g., maybe not
   hit SAVP)
   * FutureGAN
   * PixelCNN
   * VAE (Variational Autoencoder)

3. Want report to have some research attributes as well as engineering, so
   maybe also perform a short experiment? Coolest thing about video prediction
   task is that it is self-supervising, i.e., build video prediction in real
   time system?
   * Is this a good idea?
