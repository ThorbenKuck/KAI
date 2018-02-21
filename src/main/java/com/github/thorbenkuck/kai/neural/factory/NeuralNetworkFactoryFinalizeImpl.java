package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.NeuralNetwork;
import com.github.thorbenkuck.kai.neural.Neuron;

import java.util.LinkedList;
import java.util.Queue;

class NeuralNetworkFactoryFinalizeImpl implements NeuralNetworkFactoryFinalize {

	private final NNFDataObject dataObject;

	NeuralNetworkFactoryFinalizeImpl(final NNFDataObject dataObject) {
		this.dataObject = dataObject;
	}

	@Override
	public NeuralNetwork create() {
		NeuralNetworkImpl neuralNetwork = new NeuralNetworkImpl(dataObject.getInputLayer(), dataObject.getHiddenLayers(), dataObject.getOutputLayer());
		connect(neuralNetwork);
		return neuralNetwork;
	}

	private void connect(final NeuralNetworkImpl neuralNetwork) {
		final Queue<Layer> layers = new LinkedList<>(neuralNetwork.getAllLayer());

		Layer currentSource = layers.poll();

		while(layers.peek() != null) {
			Layer currentTarget = layers.poll();
			connect(currentSource, currentTarget);
			currentSource = currentTarget;
		}
	}

	private void connect(final Layer source, final Layer target) {
		for(Neuron sourceNeuron : source) {
			for(Neuron targetNeuron : target) {
				sourceNeuron.connectToOutput(targetNeuron);
				targetNeuron.connectToInput(sourceNeuron);
			}
		}
	}
}
