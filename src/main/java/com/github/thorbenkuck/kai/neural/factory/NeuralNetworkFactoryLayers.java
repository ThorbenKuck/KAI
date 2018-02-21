package com.github.thorbenkuck.kai.neural.factory;

public interface NeuralNetworkFactoryLayers {

	NeuralNetworkFactoryLayers addHiddenLayer(int numberNeurons);

	NeuralNetworkFactoryFinalize addOutputLayer(int numberNeurons);

}
