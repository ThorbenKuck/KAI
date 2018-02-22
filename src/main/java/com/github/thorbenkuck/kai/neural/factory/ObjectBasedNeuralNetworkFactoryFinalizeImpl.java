package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.neural.NeuralNetwork;
import com.github.thorbenkuck.kai.neural.Neuron;
import com.github.thorbenkuck.kai.neural.types.ShortTermMemoryNeuron;

import java.util.ArrayList;
import java.util.List;

class ObjectBasedNeuralNetworkFactoryFinalizeImpl implements ObjectBasedNeuralNetworkFactoryFinalize {

	private final ObjectBasedNNFDataObject dataObject;
	private final ObjectBasedFinalizer finalizer;

	ObjectBasedNeuralNetworkFactoryFinalizeImpl(final ObjectBasedNNFDataObject dataObject) {
		this.dataObject = dataObject;
		finalizer = new ObjectBasedFinalizer(dataObject);
	}

	@Override
	public NeuralNetwork create() {
		return finalizer.create();
	}

	@Override
	public NeuralNetworkFactoryFinalize setAllToSTMNeurons() {
		LayerImpl lastLayer = dataObject.getLastAddedLayer();
		List<Neuron> neuronList = new ArrayList<>();

		for (int i = 0; i < lastLayer.getNumberOfNeurons(); i++) {
			neuronList.add(new ShortTermMemoryNeuron());
		}

		lastLayer.applyNeurons(neuronList);

		return finalizer;
	}

	@Override
	public ObjectBasedNeuralNetworkFactoryFinalize setToSTMNeuron(final int index) {
		LayerImpl lastLayer = dataObject.getLastAddedLayer();

		if(lastLayer.getNumberOfNeurons() < index || index < 0) {
			throw new IllegalArgumentException("Cannot add neuron at: " + index + " (Layer-Height: " + lastLayer.getNumberOfNeurons() + ")");
		}

		lastLayer.setNeuron(index, new ShortTermMemoryNeuron());

		return this;
	}
}
