package com.github.thorbenkuck.kai.neural.factory;

class NeuralNetworkFactoryLayersImpl implements NeuralNetworkFactoryLayers {

	private final NNFDataObject dataObject;
	private final LayerFactory factory;

	NeuralNetworkFactoryLayersImpl(final NNFDataObject dataObject, final LayerFactory factory) {
		this.dataObject = dataObject;
		this.factory = factory;
	}

	@Override
	public NeuralNetworkFactoryLayers addHiddenLayer(final int numberNeurons) {
		dataObject.addHiddenLayer(numberNeurons);
		return this;
	}

	@Override
	public NeuralNetworkFactoryFinalize addOutputLayer(final int numberNeurons) {
		dataObject.setOutputLayer(numberNeurons);
		return new NeuralNetworkFactoryFinalizeImpl(dataObject);
	}
}
