package com.github.thorbenkuck.kai.neural.factory;

import java.util.ArrayList;
import java.util.List;

class NNFDataObject {

	private final List<Integer> hiddenLayers = new ArrayList<>();
	private int inputLayer;
	private int outputLayer;

	void addHiddenLayer(int layer) {
		hiddenLayers.add(layer);
	}

	List<Integer> getHiddenLayers() {
		return hiddenLayers;
	}

	int getInputLayer() {
		return inputLayer;
	}

	void setInputLayer(final int inputLayer) {
		this.inputLayer = inputLayer;
	}

	int getOutputLayer() {
		return outputLayer;
	}

	void setOutputLayer(final int outputLayer) {
		this.outputLayer = outputLayer;
	}
}
