package com.github.thorbenkuck.kai.neural.factory;

public interface ObjectBasedNeuralNetworkFactoryLayer {

	ObjectBasedNeuralNetworkFactoryLayer addHiddenLayer(int numberNeurons);

	ObjectBasedNeuralNetworkFactoryLayer addSTMNeuron(int... indexInLastLayer);

	ObjectBasedNeuralNetworkFactoryLayer setAllToSTMNeurons();

	ObjectBasedNeuralNetworkFactoryFinalize addOutputLayer(int numberNeurons);

}
