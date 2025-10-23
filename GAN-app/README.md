train.py: Trains a GAN (Generator and Discriminator) from scratch on the MNIST dataset (images are 28x28 and normalized to \[-1, 1])

main.py: Loads the trained model (gan\_generator.pth) and exposes a /generate endpoint to create and return a new, AI-generated handwritten digit image

generated\_images/: Stores sample images from the Generator at each epoch (500 epochs) to visualize its training progress

