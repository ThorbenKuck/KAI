package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.datatypes.QueuedIterator;
import com.github.thorbenkuck.kai.math.Matrix;
import com.github.thorbenkuck.kai.neural.Connection;
import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.Neuron;

import java.util.*;

class LayerImpl implements Layer {

	private final List<Neuron> neurons;
	private Matrix lastGuess = new Matrix(0);
	private Layer nextLayer;
	private Layer previousLayer;

	LayerImpl(List<Neuron> neuronList) {
		neurons = new ArrayList<>(neuronList);
	}

	/**
	 * Returns an iterator over elements of type {@code T}.
	 *
	 * @return an Iterator.
	 */
	@Override
	public Iterator<Neuron> iterator() {
		return new QueuedIterator<>(neurons);
	}

	@Override
	public int getNumberOfNeurons() {
		return neurons.size();
	}

	@Override
	public Neuron getNeuronAtIndex(final int index) {
		return neurons.get(index);
	}

	@Override
	public void setNeuron(final int index, final Neuron neuron) {
		neurons.set(index, neuron);
	}

	@Override
	public void randomizeInputConnectionWeights() {
		for (Neuron neuron : neurons) {
			for (Connection connection : neuron.getAllInputConnections()) {
				connection.setWeight(Connection.randomWeight());
				connection.setDeltaWeight(Connection.randomWeight());
			}
			for (Connection connection : neuron.getAllOutputConnections()) {
				connection.setWeight(Connection.randomWeight());
				connection.setDeltaWeight(Connection.randomWeight());
			}
		}
	}

	@Override
	public void randomizeOutputConnectionWeights() {
		for (Neuron neuron : neurons) {
			for (Connection connection : neuron.getAllOutputConnections()) {
				connection.setWeight(Connection.randomWeight());
				connection.setDeltaWeight(Connection.randomWeight());
			}
		}
	}

	@Override
	public Matrix guess() {
		List<Double> result = new ArrayList<>();
		for (Neuron neuron : neurons) {
			result.add(neuron.guess());
		}

		lastGuess = Matrix.fromArray(result.toArray(new Double[result.size()]));
		return lastGuess;
	}

	@Override
	public Matrix getOutputs() {
		List<Double> result = new ArrayList<>();
		for (Neuron neuron : neurons) {
			result.add(neuron.getOutputValue());
		}

		return Matrix.fromArray(result.toArray(new Double[result.size()]));
	}

	@Override
	public void setInputData(Double[] data) {
		for (int i = 0; i < data.length; i++) {
			Double currentInput = data[i];
			Neuron neuron = neurons.get(i);
			neuron.setInputValue(currentInput);
		}
	}

	@Override
	public boolean isInput() {
		for (Neuron neuron : neurons) {
			if (!neuron.isInput()) {
				return false;
			}
		}

		return true;
	}

	@Override
	public boolean isOutput() {
		for (Neuron neuron : neurons) {
			if (!neuron.isOutput()) {
				return false;
			}
		}

		return true;
	}

	@Override
	public boolean isHidden() {
		return ! isInput() && ! isOutput();
	}

	@Override
	public Layer copy() {
		return new LayerImpl(new ArrayList<>(neurons));
	}

	@Override
	public Layer getNext() {
		return nextLayer;
	}

	@Override
	public Layer getPrevious() {
		return previousLayer;
	}

	@Override
	public void setNext(final Layer next) {
		this.nextLayer = next;
	}

	@Override
	public void setPrevious(final Layer previous) {
		this.previousLayer = previous;
	}

	@Override
	public List<Neuron> copyAllNeurons() {
		return Collections.unmodifiableList(neurons);
	}

	@Override
	public void resetError() {
		for(Neuron neuron : neurons) {
			neuron.setError(0.0);
		}
	}

	@Override
	public String toString() {
		return " LayerImpl {" + "neurons =" + neurons + '}';
	}

	void applyNeurons(List<Neuron> neurons) {
		this.neurons.clear();
		this.neurons.addAll(neurons);
	}
}
