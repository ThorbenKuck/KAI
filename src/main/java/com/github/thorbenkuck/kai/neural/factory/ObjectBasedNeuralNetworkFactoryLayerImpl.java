package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.neural.Neuron;
import com.github.thorbenkuck.kai.neural.types.ShortTermMemoryNeuron;

import java.util.ArrayList;
import java.util.List;

public class ObjectBasedNeuralNetworkFactoryLayerImpl implements ObjectBasedNeuralNetworkFactoryLayer {

	private final ObjectBasedNNFDataObject dataObject;
	private final LayerFactory layerFactory;

	public ObjectBasedNeuralNetworkFactoryLayerImpl(final ObjectBasedNNFDataObject dataObject, final LayerFactory layerFactory) {
		this.dataObject = dataObject;
		this.layerFactory = layerFactory;
	}

	@Override
	public ObjectBasedNeuralNetworkFactoryLayer addHiddenLayer(final int numberNeurons) {
		LayerImpl hiddenLayer = layerFactory.apply(numberNeurons);
		dataObject.addHiddenLayer(hiddenLayer);
		return this;
	}

	@Override
	public ObjectBasedNeuralNetworkFactoryLayer addSTMNeuron(final int... indexes) {
		LayerImpl toWorkOn = dataObject.getLastAddedLayer();
		for(int indexInLastLayer : indexes) {
			if (indexInLastLayer >= toWorkOn.getNumberOfNeurons() || indexInLastLayer < 0) {
				throw new IllegalArgumentException("Cannot add neuron at: " + indexInLastLayer + " (Layer-Height: " + toWorkOn.getNumberOfNeurons() + ")");
			}
			toWorkOn.setNeuron(indexInLastLayer, new ShortTermMemoryNeuron());
		}
		return this;
	}

	@Override
	public ObjectBasedNeuralNetworkFactoryLayer setAllToSTMNeurons() {
		LayerImpl toWorkOn = dataObject.getLastAddedLayer();
		List<Neuron> newNeurons = new ArrayList<>();
		for(int i = 0; i < toWorkOn.getNumberOfNeurons() ; i++) {
			newNeurons.set(i, new ShortTermMemoryNeuron());
		}
		toWorkOn.applyNeurons(newNeurons);
		return this;
	}

	@Override
	public ObjectBasedNeuralNetworkFactoryFinalize addOutputLayer(final int numberNeurons) {
		LayerImpl outputLayer = layerFactory.apply(numberNeurons);
		dataObject.setOutputLayer(outputLayer);
		return new ObjectBasedNeuralNetworkFactoryFinalizeImpl(dataObject);
	}
}
