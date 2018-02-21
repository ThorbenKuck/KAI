package com.github.thorbenkuck.kai.neural.factory;

class NeuralNetworkFactoryImpl implements NeuralNetworkFactory {

	private final NNFDataObject dataObject = new NNFDataObject();
	private final LayerFactory layerFactory = new LayerFactory();

	@Override
	public NeuralNetworkFactoryLayers addInputLayer(final int inputNeurons) {
		dataObject.setInputLayer(inputNeurons);
		return new NeuralNetworkFactoryLayersImpl(dataObject, layerFactory);
	}
}
