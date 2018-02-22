package com.github.thorbenkuck.kai.neural.factory;

class MatrixBasedNeuralNetworkFactoryLayersImpl implements NeuralNetworkFactoryLayers {

	private final MatrixBasedNNFDataObject dataObject;

	MatrixBasedNeuralNetworkFactoryLayersImpl(final MatrixBasedNNFDataObject dataObject) {
		this.dataObject = dataObject;
	}

	@Override
	public NeuralNetworkFactoryLayers addHiddenLayer(final int numberNeurons) {
		dataObject.addHiddenLayer(numberNeurons);
		return this;
	}

	@Override
	public NeuralNetworkFactoryFinalize addOutputLayer(final int numberNeurons) {
		dataObject.setOutputLayer(numberNeurons);
		return new MatrixBasedNeuralNetworkFactoryFinalizeImpl(dataObject);
	}
}
