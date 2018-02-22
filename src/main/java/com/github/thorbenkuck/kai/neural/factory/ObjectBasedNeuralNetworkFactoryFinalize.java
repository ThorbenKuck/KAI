package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.neural.NeuralNetwork;

public interface ObjectBasedNeuralNetworkFactoryFinalize {

	NeuralNetwork create();

	NeuralNetworkFactoryFinalize setAllToSTMNeurons();

	ObjectBasedNeuralNetworkFactoryFinalize setToSTMNeuron(int index);

}
