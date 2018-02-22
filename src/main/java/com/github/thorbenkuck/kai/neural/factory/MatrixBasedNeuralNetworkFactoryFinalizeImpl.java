package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.neural.NeuralNetwork;

class MatrixBasedNeuralNetworkFactoryFinalizeImpl implements NeuralNetworkFactoryFinalize {

	private final MatrixBasedNNFDataObject dataObject;

	MatrixBasedNeuralNetworkFactoryFinalizeImpl(final MatrixBasedNNFDataObject dataObject) {
		this.dataObject = dataObject;
	}

	@Override
	public NeuralNetwork create() {
		return new MatrixBasedNeuralNetwork(dataObject.getInputLayer(), dataObject.getHiddenLayers(), dataObject.getOutputLayer());
	}
}
