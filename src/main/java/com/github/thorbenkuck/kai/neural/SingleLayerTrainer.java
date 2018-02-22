package com.github.thorbenkuck.kai.neural;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class SingleLayerTrainer {

	private double learningRate = 0.1;
	private double maxError = 0.000000000001;
	private long probingTime = 1000;
	private final AtomicBoolean running = new AtomicBoolean(false);

	public void train(List<Neuron> outputs, List<Double[]> inputData, List<Double[]> expectedOutputData) {
		if (inputData.size() != expectedOutputData.size()) {
			throw new IllegalArgumentException("Data-Sets do not match");
		}

		double error = 1;
		long currentIteration = 0;
		running.set(true);
		while (error > maxError && running.get()) {
			error = 0;
			for (int i = 0; i < inputData.size(); i++) {
				double localError = 0;
				Double[] inputDataArray = inputData.get(i);
				Double[] expectedDataArray = expectedOutputData.get(i);
				for (int j = 0; j < expectedDataArray.length; j++) {
					Neuron outputNeuron = outputs.get(j);

					localError += outputNeuron.learn(inputDataArray, expectedDataArray[j], 0.1);
				}
				error += localError;
			}
			if(currentIteration % probingTime == 0) {
				System.out.println("Error after " + currentIteration + " trainings cycles: " + error);
			}
			++currentIteration;
		}
		running.set(false);
	}

	public void stop() {
		running.set(false);
	}
}
