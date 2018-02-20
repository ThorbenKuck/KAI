package com.github.thorbenkuck.kai.neural;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetworkCalibrator {

	private double epsilon = 0.0000000000001;
	private double learningRate = 0.9f;
	private double momentum = 0.7f;

	public double learn(List<Double[]> inputData, List<Double[]> outputData, Layer inputLayer, List<Layer> hiddenLayers, Layer outputLayer) {
		double error = 0.0;

		for (int i = 0; i < inputData.size(); i++) {
			calibrateInputs(inputLayer, inputData.get(i));
			outputLayer.calculate();
			final List<Double> expectedOutputData = Arrays.asList(outputData.get(i));
			calibrateOutputs(outputLayer, expectedOutputData);
			calibrateHiddenLayers(outputLayer, hiddenLayers, expectedOutputData);
			List<Double> calculatedOutputs = getOutputs(outputLayer);
			for(int j = 0 ; j < expectedOutputData.size() ; j++) {
				double expected = expectedOutputData.get(j);
				double calculated = calculatedOutputs.get(i);

				error = error + (expected - calculated);
			}
		}

		return error;
	}

	private void calibrateOutputs(Layer outputLayer, List<Double> expectedResults) {
		for(int i = 0 ; i < outputLayer.size() ; i++) {
			Neuron neuron = outputLayer.get(i);
			double expected = expectedResults.get(i);
			double calculated = neuron.getOutput();

			for(Connection connection : neuron.getAllInputConnections()) {
				double prevCalculated = connection.getSource().getOutput();
				double partialDerivative = -calculated * (1 - calculated) * prevCalculated
						* (expected - calculated);
				double newDeltaWeight = -learningRate * partialDerivative;
				double newWeight = connection.getWeight() + connection.getDeltaWeight();

				connection.setWeight(newWeight + momentum * connection.getDeltaWeight());
				connection.setDeltaWeight(newDeltaWeight);
			}
		}
	}

	private void calibrateHiddenLayers(Layer outputLayer, List<Layer> hiddenLayers, List<Double> expectedOutputData) {
		if (hiddenLayers.isEmpty()) {
			return;
		}

		for (Layer layer : hiddenLayers) {
			for (int i = 0; i < layer.size(); i++) {
				final Neuron neuron = layer.get(i);
				double expected = expectedOutputData.get(i);
				double calculated = (neuron.getOutput());

				for (Connection connection : neuron.getAllInputConnections()) {
					double prevCalculated = connection.getSource().getOutput();

					double outputSum = 0;
					for (Neuron outputNeuron : outputLayer) {
						double wjk = outputNeuron.getConnectionTo(neuron).getWeight();
						double outputCalculated = outputNeuron.getOutput();

						outputSum += ((-expected - outputCalculated) * outputCalculated * (1 - outputCalculated) * wjk);
					}

					double partialDerivative = -calculated * (1 - calculated) * prevCalculated
							* outputSum;
					double newDeltaWeight = -learningRate * partialDerivative;
					double newWeight = connection.getWeight() + connection.getDeltaWeight();

					connection.setWeight(newWeight + momentum * connection.getDeltaWeight());
					connection.setDeltaWeight(newDeltaWeight);
				}
			}
		}
	}

	private void calibrateInputs(Layer layer, Double[] dataSet) {
		for (int i = 0; i < layer.size(); i++) {
			Neuron neuron = layer.get(i);
			double data = dataSet[i];
			neuron.setInput(data);
		}
	}

	private List<Double> getOutputs(Layer outputLayer) {
		List<Double> calculated = new ArrayList<>();
		for(Neuron neuron : outputLayer) {
			calculated.add(neuron.getOutput());
		}
		return calculated;
	}

}
