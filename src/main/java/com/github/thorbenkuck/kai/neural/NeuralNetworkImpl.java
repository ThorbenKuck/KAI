package com.github.thorbenkuck.kai.neural;

import java.util.*;

class NeuralNetworkImpl implements NeuralNetwork {

	private final Layer inputLayer;
	private final Layer outputLayer;
	private final List<Layer> hiddenLayers = new ArrayList<>();
	private final NeuralNetworkCalibrator calibrator = new NeuralNetworkCalibrator();
	private int maxIterations = 10000;
	private double maxError = 0.01;

	public NeuralNetworkImpl(Layer inputLayer, Layer outputLayer) {
		this(inputLayer, new ArrayList<>(), outputLayer);
	}

	public NeuralNetworkImpl(Layer inputLayer, List<Layer> hiddenLayers, Layer outputLayer) {
		this.inputLayer = inputLayer;
		this.outputLayer = outputLayer;
		this.hiddenLayers.addAll(hiddenLayers);
		calibrate();
	}

	private void calibrate() {
		if (hiddenLayers.isEmpty()) {
			calibrateOnlyInputAndOutput();
		} else {
			calibrateAll();
		}
	}

	private void calibrateAll() {
		final Queue<Layer> allLayers = new LinkedList<>();
		allLayers.add(inputLayer);
		allLayers.addAll(hiddenLayers);
		allLayers.add(outputLayer);
		calibrateQueue(allLayers);
	}

	private void calibrateQueue(final Queue<Layer> queue) {
		Layer inputLayer = queue.poll();
		Layer outputLayer = queue.poll();

		if (queue.isEmpty()) {
			connect(inputLayer, outputLayer);
			return;
		}

		while (queue.peek() != null) {
			connect(inputLayer, outputLayer);
			inputLayer = outputLayer;
			outputLayer = queue.poll();
		}
		connect(inputLayer, outputLayer);
	}

	private void calibrateOnlyInputAndOutput() {
		connect(inputLayer, outputLayer);
	}

	private void connect(Layer sourceLayer, Layer targetLayer) {
		for (Neuron inputNeuron : sourceLayer) {
			for (Neuron outputNeuron : targetLayer) {
				connect(inputNeuron, outputNeuron);
			}
		}
	}

	private void connect(Neuron inputNeuron, Neuron outputNeuron) {
		inputNeuron.connectToOutput(outputNeuron);
		outputNeuron.connectToInput(inputNeuron);
	}

	@Override
	public boolean train(List<Double[]> inputData, List<Double[]> outputData) {
		if (inputData.size() != outputData.size()) {
			throw new IllegalArgumentException("Unequal Training set! " + inputData + " | " + outputData);
		}

		double error;
		int iteration = 0;

		do {
			error = calibrator.learn(inputData, outputData, inputLayer, hiddenLayers, outputLayer);
			System.out.println("(" + iteration + ")Error: " + error);
			++iteration;
			if (iteration >= maxIterations) {
				break;
			}
		} while (error > maxError);

		return error < maxError;
	}

	@Override
	public List<Double> calculate(Double[] inputData) {
		return calculate(Arrays.asList(inputData));
	}

	@Override
	public List<Double> calculate(List<Double> inputData) {
		if (inputData.size() != inputLayer.size()) {
			throw new IllegalArgumentException("Data-sizes do not match! Provided: " + inputData.size() + " data points. Input Neurons: " + inputLayer.size());
		}

		setInputs(inputData);
		return calculateOutputs();
	}

	private List<Double> calculateOutputs() {
		final List<Double> results = new ArrayList<>();

		for (Neuron neuron : outputLayer) {
			neuron.calculate();
		}
		for (Neuron neuron : outputLayer) {
			results.add(neuron.getOutput());
		}

		return results;
	}

	private void setInputs(List<Double> inputData) {
		for (int i = 0; i < inputLayer.size(); i++) {
			Neuron neuron = inputLayer.get(i);
			neuron.setInput(inputData.get(i));
		}
	}

	@Override
	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}
}
