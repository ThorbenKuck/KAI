package com.github.thorbenkuck.kai.neural;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;

public class NeuralNetworkTrainer {

	public static final BiConsumer<NeuralNetwork, NeuralNetworkTrainer> DEFAULT_ERROR_GROWTH_CONSUMER = new DefaultErrorConsumer();
	public static final BiConsumer<NeuralNetwork, NeuralNetworkTrainer> DEFAULT_FAILED_TRAINING_CONSUMER = new SecondChanceStrategy();
	public static final BiConsumer<NeuralNetwork, NeuralNetworkTrainer> DEFAULT_PROBING_CONSUMER = new DefaultProbingConsumer();
	private final List<Double[]> trainingInputData = new ArrayList<>();
	private final List<Double[]> trainingOutputData = new ArrayList<>();
	private long maxIterations = 10000000;
	private final AtomicBoolean running = new AtomicBoolean(false);
	private double maximumError = 0.00000000001;
	private int probingIndex = 1000;
	private double calculatedError = 0;
	private double lastCalculatedError = 0;
	private int currentIteration = 0;
	private double tolerance = 0.01;
	private BiConsumer<NeuralNetwork, NeuralNetworkTrainer> errorGrowthConsumer = DEFAULT_ERROR_GROWTH_CONSUMER;
	private BiConsumer<NeuralNetwork, NeuralNetworkTrainer> failedTrainingConsumer = DEFAULT_FAILED_TRAINING_CONSUMER;
	private BiConsumer<NeuralNetwork, NeuralNetworkTrainer> probingConsumer = DEFAULT_PROBING_CONSUMER;
	private int stagnationCount = 0;

	private double trainWithAllData(NeuralNetwork neuralNetwork) {
		double calculatedError = 0;
		for (int i = 0; i < trainingInputData.size(); i++) {
			Double[] input = trainingInputData.get(i);
			Double[] output = trainingOutputData.get(i);
			calculatedError += neuralNetwork.train(input, output);
		}

		// Todo, ist das 0.0?
		return calculatedError / trainingInputData.size();
	}

	public double getMaximumError() {
		return maximumError;
	}

	public double getCurrentCalculatedError() {
		return calculatedError;
	}

	public double getLastCalculatedError() {
		return lastCalculatedError;
	}

	public int getCurrentIteration() {
		return currentIteration;
	}

	public void setCurrentIteration(int to) {
		this.currentIteration = to;
	}

	public void setMaxIterations(int max) {
		this.maxIterations = max;
	}

	public void setMaximumError(double error) {
		this.maximumError = error;
	}

	public void setProbingTime(int iterationCycle) {
		this.probingIndex = iterationCycle;
	}

	public void setAmountOfProbing(int howOftenToProbe) {
		if(this.maxIterations != -1) {
			setProbingTime(Math.round(maxIterations / howOftenToProbe));
		} else {
			setProbingTime(10000);
		}
	}

	public void stop() {
		running.set(false);
	}

	public void setTrainingInputData(List<Double[]> data) {
		this.trainingInputData.clear();
		this.trainingInputData.addAll(data);
	}

	public void setTrainingOutputData(List<Double[]> data) {
		this.trainingOutputData.clear();
		this.trainingOutputData.addAll(data);
	}

	public long getMaximumTrainingCycles() {
		return maxIterations;
	}

	public void reset() {
		calculatedError = 0;
		lastCalculatedError = 0;
		currentIteration = 0;
		stop();
	}

	public void setIndefiniteTrainingEnd() {
		maxIterations = -1;
	}

	public void setTolerance(double errorTolerance) {
		this.tolerance = errorTolerance;
	}

	public void setProbingConsumer(BiConsumer<NeuralNetwork, NeuralNetworkTrainer> consumer) {
		this.probingConsumer = consumer;
	}

	public void train(NeuralNetwork neuralNetwork) {
		running.set(true);
		if(trainingInputData.size() != trainingOutputData.size()) {
			throw new IllegalStateException("Provided input data and output data do not match!");
		}
		do {

			calculatedError = trainWithAllData(neuralNetwork);

			if (currentIteration % probingIndex == 0) {
				probingConsumer.accept(neuralNetwork, this);
				if(lastCalculatedError < calculatedError) {
					errorGrowthConsumer.accept(neuralNetwork, this);
				}
			}
			++ currentIteration;
			lastCalculatedError = calculatedError;
		} while ((calculatedError > maximumError && ! maxIterationsReached()) && running.get());

		if(running.get() && calculatedError > maximumError) {
			failedTrainingConsumer.accept(neuralNetwork, this);
		}

		running.set(false);
	}

	private boolean maxIterationsReached() {
		return maxIterations != -1 && currentIteration >= maxIterations;
	}

	private static class DefaultErrorConsumer implements BiConsumer<NeuralNetwork, NeuralNetworkTrainer> {

		@Override
		public void accept(final NeuralNetwork neuralNetwork, final NeuralNetworkTrainer neuralNetworkTrainer) {
			System.out.println("Growth in error by " + (neuralNetworkTrainer.lastCalculatedError - neuralNetworkTrainer.calculatedError) + " detected!");
		}
	}

	private static class SecondChanceStrategy implements BiConsumer<NeuralNetwork, NeuralNetworkTrainer> {

		/**
		 * Performs this operation on the given arguments.
		 *
		 * @param neuralNetwork        the first input argument
		 * @param neuralNetworkTrainer the second input argument
		 */
		@Override
		public void accept(final NeuralNetwork neuralNetwork, final NeuralNetworkTrainer neuralNetworkTrainer) {
			if(! resultsAreAcceptable(neuralNetwork, neuralNetworkTrainer)) {
				System.out.println("X");
				System.out.println("Second chance not applicable. Restart training!");
				reset(neuralNetwork, neuralNetworkTrainer);
			}
		}

		private boolean resultsAreAcceptable(NeuralNetwork neuralNetwork, NeuralNetworkTrainer trainer) {
			System.out.println("Looking for a second chance..");
			for(int i = 0 ; i < trainer.trainingInputData.size() ; i++) {
				Double[] inputData = trainer.trainingInputData.get(i);
				Double[] outputData = trainer.trainingOutputData.get(i);
				Double[] calculated = neuralNetwork.feedForward(inputData);
				System.out.println(Arrays.toString(inputData) + " => " + Arrays.toString(calculated) + " ( " + Arrays.toString(outputData) + " + " + trainer.tolerance + " )");
				for(int j = 0 ; j < outputData.length ; j++) {
					Double expectedData = outputData[j];
					Double calculatedData = calculated[j];
					if(!(Math.abs(expectedData - calculatedData) < trainer.tolerance)) {
						return false;
					}
				}
			}
			return true;
		}
	}

	static void reset(NeuralNetwork neuralNetwork, NeuralNetworkTrainer trainer) {
		System.out.println("Resetting...");
		neuralNetwork.reset();
		neuralNetwork.setLearningRate(neuralNetwork.getLearningRate() * 0.95);
		trainer.stop();
		trainer.reset();
		trainer.train(neuralNetwork);
	}

	private static class DefaultProbingConsumer implements BiConsumer<NeuralNetwork, NeuralNetworkTrainer> {

		/**
		 * Performs this operation on the given arguments.
		 *
		 * @param neuralNetwork        the first input argument
		 */
		@Override
		public void accept(final NeuralNetwork neuralNetwork, final NeuralNetworkTrainer trainer) {
			System.out.println("error after (" + trainer.currentIteration + "/" + trainer.maxIterations + ") trainings-cycles: " + trainer.calculatedError);
			if ((trainer.calculatedError * 2) < trainer.lastCalculatedError) {
				System.out.println("Big jump in effectiveness of Neural Network detected! Resetting iteration count");
				trainer.currentIteration = 0;
			}

			if(trainer.calculatedError == trainer.lastCalculatedError) {
				System.out.println("STAGNATION DETECTED!");
				++trainer.stagnationCount;
				if(trainer.stagnationCount >= 10) {
					System.out.println("reset imminent");
					reset(neuralNetwork, trainer);
				}
			} else {
				trainer.stagnationCount = 0;
			}
		}
	}
}
