# Neuro-Evolution
Custom library of NeuroEvolution of Augmenting Topologies.

Based on the work of Kenneth Stanley and Risto Miikkulainen. [[1]](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf)

## What is it?
This repo is my implementation of Stanley and Miikkulainen's NEAT algorithm. 
It is a genetic algorithm for evolving and training neural networks.

## Why use NEAT over traditional training?
In traditional training of neural networks, the architecture is typically choosen by hand. 

The architecture of a neural network can include the following features:
  - The number of deep layers.
  - The number of neurons in a deep layer
  - The number of connection weights between neurons inbetween the layers

Then, the weights between each neuron are adjusted based on algorithms such as gradient descent.

NEAT takes a different approach. Through a genetic algorithm over multiple generations, neural networks are not only able to find the weights on the fly, but also the ideal architecture as well. If set up well, this approach can be great, especially if you are unsure about the best architecture that suits your environment.

## Disclaimer:

As of writing this README.md (5/9/2023), this library does not implement protecting innovation through speciations.
This is a fundamental concept introduced by Stanley, as it gives newly mutated genomes time to optimize their architectures.

However, this is a major TODO feature for later.
