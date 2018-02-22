package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.math.Matrix;
import com.github.thorbenkuck.kai.neural.Connection;
import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.NeuralNetwork;
import com.github.thorbenkuck.kai.neural.Neuron;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

class ObjectBasedNeuralNetwork implements NeuralNetwork {

	private final List<Layer> hiddenLayers = new ArrayList<>();
	private Layer inputLayer;
	private Layer outputLayer;
	private double learningRate = 0.1;

	ObjectBasedNeuralNetwork(Layer inputLayer, Layer outputLayer) {
		this(inputLayer, new ArrayList<>(), outputLayer);
	}

	ObjectBasedNeuralNetwork(Layer inputLayer, List<Layer> hiddenLayers, Layer outputLayer) {
		this.inputLayer = inputLayer;
		this.hiddenLayers.addAll(hiddenLayers);
		this.outputLayer = outputLayer;
	}

	private List<Matrix> feedForward() {
		List<Matrix> results = new ArrayList<>();
		results.add(inputLayer.guess());
		for(Layer layer : hiddenLayers) {
			results.add(layer.guess());
		}
		results.add(outputLayer.guess());

		return results;
	}

	private void setInputData(final Double[] input) {
		if (input.length != inputLayer.getNumberOfNeurons()) {
			throw new IllegalArgumentException("Input data (" + input.length + ") does not match the number of input neurons(" + inputLayer.getNumberOfNeurons() + ")");
		}

		inputLayer.setInputData(input);
	}

	@Override
	public Double[] evaluate(final Double[] input) {
		setInputData(input);
		feedForward();

		Matrix result = outputLayer.guess();

		return result.to1DArray();
	}

	@Override
	public double train(final Double[] inputs, final Double[] answer) {
		setInputData(inputs);
		List<Matrix> results = feedForward();
		final Matrix outputResult = results.get(results.size() - 1);
		final Matrix expectedResults = Matrix.fromArray(answer);
		// FEHLER = ist-ausagbe - sollausgabe
		final Matrix outputError = Matrix.subtract(outputResult, expectedResults);

		// Clear out existing matrices to free up memory
		outputResult.clear();
		expectedResults.clear();
		for(Matrix matrix : results) {
			matrix.clear();
		}

		List<Layer> layers = new ArrayList<>(getAllLayer());
		Collections.reverse(layers);

		for(Layer layer : layers) {
			layer.resetError();
		}

		double errorSum = 0.0;
		for(Layer layer : layers) {
			errorSum = trainLayer(layer, outputError);
		}

		return outputError.absoluteSum();
	}

	private double trainLayer(Layer layer, Matrix error) {
		if(layer.isOutput()) {
			return trainOutputLayer(layer, error);
		} else {
			return trainHiddenLayer(layer);
		}
	}

	private double trainHiddenLayer(final Layer layer) {
		// Weight = Weight + deltaWeight
		double totalError = 0;
		for(int i = 0 ; i < layer.getNumberOfNeurons() ; i++) {
			Neuron neuron = layer.getNeuronAtIndex(i);
			double error = 0.0;
			// Summe von fehler aller vorherigen * gewicht an diese.
			for(Connection connection : neuron.getAllOutputConnections()) {
				error += connection.getTarget().getError() * connection.getWeight();
			}
			totalError += error;
			neuron.setError(error);
			for(Connection connection : neuron.getAllInputConnections()) {
				trainNeuron(neuron, connection);
			}
		}
		return totalError / layer.getNumberOfNeurons();
	}

	private double trainOutputLayer(Layer layer, Matrix errorMatrix) {
		for(int i = 0 ; i < layer.getNumberOfNeurons() ; i++) {
			Neuron neuron = layer.getNeuronAtIndex(i);
			double error = errorMatrix.getPoint(i, 0);
			neuron.setError(error);
			for(Connection connection : neuron.getAllInputConnections()) {
				trainNeuron(neuron, connection);
			}
		}
		return errorMatrix.absoluteSum() / (errorMatrix.getRows() * errorMatrix.getColumns());
	}

	private void trainNeuron(Neuron neuron, Connection ingoingConnection) {
		// deltaWeight = -lernrate * (abl(aktivierungsfunktion) * FEHLER) * ausgabeVonVorherigemNeuron
		double weightedError = (neuron.getActivationFunction().calculateDerivative(neuron.getOutputValue())) *(neuron.getError());
		double deltaWeight = -learningRate * weightedError * ingoingConnection.getSource().getOutputValue();
		ingoingConnection.addToWeight(deltaWeight);
	}

	@Override
	public void reset() {
		outputLayer.randomizeInputConnectionWeights();
		for (Layer layer : hiddenLayers) {
			layer.randomizeInputConnectionWeights();
		}
	}

	@Override
	public void setLearningRate(final double to) {
		this.learningRate = to;
	}

	@Override
	public double getLearningRate() {
		return learningRate;
	}

	@Override
	public String toString() {
		return "ObjectBasedNeuralNetwork{" +
				"inputLayer=" + inputLayer +
				", hiddenLayers=" + hiddenLayers +
				", outputLayer=" + outputLayer +
				'}';
	}

	@Override
	public String toPrettyString() {
		StringBuilder stringBuilder = new StringBuilder();

		for(Neuron neuron : inputLayer) {
			printNeuron(neuron, stringBuilder);
		}

		return stringBuilder.toString();
	}

	private void printNeuron(Neuron neuron, StringBuilder stringBuilder) {
		for(Connection connection : neuron.getAllOutputConnections()) {
			stringBuilder.append(neuron.toString()).append("--").append(connection.getWeight()).append("-->");
			printNeuron(connection.getTarget(), stringBuilder);
			stringBuilder.append(System.lineSeparator());
		}
	}

	// Package private for factory
	Collection<Layer> getAllLayer() {
		final List<Layer> layers = new ArrayList<>();
		layers.add(inputLayer);
		layers.addAll(hiddenLayers);
		layers.add(outputLayer);

		return Collections.unmodifiableCollection(layers);
	}
}