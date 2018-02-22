package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.NeuralNetwork;
import com.github.thorbenkuck.kai.neural.Neuron;

import java.util.LinkedList;
import java.util.Queue;

class ObjectBasedFinalizer implements NeuralNetworkFactoryFinalize {

	private final ObjectBasedNNFDataObject dataObject;

	ObjectBasedFinalizer(final ObjectBasedNNFDataObject dataObject) {
		this.dataObject = dataObject;
	}

	private void connect(final ObjectBasedNeuralNetwork neuralNetwork) {
		final Queue<Layer> layers = new LinkedList<>(neuralNetwork.getAllLayer());

		Layer currentSource = layers.poll();

		while(layers.peek() != null) {
			Layer currentTarget = layers.poll();
			connect(currentSource, currentTarget);
			currentSource.setNext(currentTarget);
			currentTarget.setPrevious(currentSource);
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

	@Override
	public NeuralNetwork create() {
		ObjectBasedNeuralNetwork neuralNetwork = new ObjectBasedNeuralNetwork(dataObject.getInputLayer(), dataObject.getHiddenLayers(), dataObject.getOutputLayer());
		connect(neuralNetwork);
		return neuralNetwork;
	}
}
