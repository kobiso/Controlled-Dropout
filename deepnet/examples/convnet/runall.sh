#!/bin/bash
# Trains a feed forward net on MNIST.
train_deepnet='python ../../trainer.py'
${train_deepnet} cifar10.pbtxt train.pbtxt eval.pbtxt
