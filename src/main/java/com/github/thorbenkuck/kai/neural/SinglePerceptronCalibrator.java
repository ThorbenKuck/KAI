package com.github.thorbenkuck.kai.neural;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SinglePerceptronCalibrator {

	public void train(List<Neuron> inputs, List<Neuron> outputs, List<Double[]> inputData, List<Double[]> expectedOutputData) {
		if(inputData.size() != expectedOutputData.size()) {
			throw new IllegalArgumentException("Data-Sets do not match");
		}

		double error = 1;

		while(error > 0.00001) {
			error = 0;
			for (int i = 0; i < inputData.size(); i++) {
				double localError = 0;
				Double[] inputDataArray = inputData.get(i);
				Double[] expectedDataArray = expectedOutputData.get(i);
				for(int j = 0 ; j < expectedDataArray.length ; j++) {
					Neuron outputNeuron = outputs.get(j);
					localError += outputNeuron.learn(inputDataArray, expectedDataArray[j], 0.1);
				}
				error += localError;
			}
			error /= expectedOutputData.size();
			System.out.println(error);
		}
	}

	private void setInputs(List<Neuron> inputs, Double[] inputData) {
		if(inputs.size() != inputData.length) {
			throw new IllegalArgumentException("Could not match " + Arrays.toString(inputData) + " to " + inputs);
		}

		for(int i = 0 ; i < inputs.size() ; i++) {
			Neuron neuron = inputs.get(i);
			neuron.setInput(inputData[i]);
		}
	}

	private Double[] getCalculatedOutputs(List<Neuron> outputs) {
		List<Double> outputData = new ArrayList<>();

		for(Neuron neuron : outputs) {
			neuron.calculate();
			outputData.add(neuron.getOutput());
		}

		return outputData.toArray(new Double[outputData.size()]);
	}

	private void adapt(final List<Neuron> outputs, final Double[] expectedDataArray) {
		for(int i = 0 ; i < outputs.size() ; i++) {
			Neuron neuron = outputs.get(i);
			double data = expectedDataArray[i];
			neuron.correct(data);
		}
	}

}
