package com.github.thorbenkuck.kai.neural.factory;

class MatrixBasedNeuralNetworkFactoryImpl implements MatrixBasedNeuralNetworkFactory {

	private final MatrixBasedNNFDataObject dataObject = new MatrixBasedNNFDataObject();

	@Override
	public NeuralNetworkFactoryLayers addInputLayer(final int inputNeurons) {
		dataObject.setInputLayer(inputNeurons);
		return new MatrixBasedNeuralNetworkFactoryLayersImpl(dataObject);
	}
}
