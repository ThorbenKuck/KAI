package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.neural.Layer;

import java.util.ArrayList;
import java.util.List;

class ObjectBasedNNFDataObject {

	private LayerImpl inputLayer;
	private final List<LayerImpl> hiddenLayers = new ArrayList<>();
	private LayerImpl outputLayer;
	private LayerImpl pointer;

	private void setPointer(LayerImpl layer) {
		pointer = layer;
	}

	public LayerImpl getInputLayer() {
		return inputLayer;
	}

	public void setInputLayer(final LayerImpl inputLayer) {
		this.inputLayer = inputLayer;
		setPointer(inputLayer);
	}

	public void addHiddenLayer(final LayerImpl hiddenLayer) {
		this.hiddenLayers.add(hiddenLayer);
		setPointer(hiddenLayer);
	}

	public List<Layer> getHiddenLayers() {
		return new ArrayList<>(hiddenLayers);
	}

	public LayerImpl getOutputLayer() {
		return outputLayer;
	}

	public void setOutputLayer(final LayerImpl outputLayer) {
		this.outputLayer = outputLayer;
		setPointer(outputLayer);
	}

	public LayerImpl getLastAddedLayer() {
		return pointer;
	}
}
