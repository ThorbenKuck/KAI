package com.github.thorbenkuck.kai.neural.factory;

class ObjectBasedNeuralNetworkFactoryImpl implements ObjectBasedNeuralNetworkFactory {

	private final LayerFactory layerFactory = new LayerFactory();
	private final ObjectBasedNNFDataObject dataObject = new ObjectBasedNNFDataObject();

	@Override
	public ObjectBasedNeuralNetworkFactoryLayer addInputLayer(final int numberNeurons) {
		LayerImpl inputLayer = layerFactory.createSupplierLayer(numberNeurons);
		dataObject.setInputLayer(inputLayer);
		return new ObjectBasedNeuralNetworkFactoryLayerImpl(dataObject, layerFactory);
	}
}
