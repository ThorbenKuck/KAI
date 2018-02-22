package com.github.thorbenkuck.kai;

import com.github.thorbenkuck.kai.neural.*;
import com.github.thorbenkuck.kai.neural.factory.NeuralNetworkFactory;
import com.github.thorbenkuck.kai.neural.types.AbstractNeuron;
import com.github.thorbenkuck.kai.neural.types.SupplierNeuron;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class Example {

	private static final List<Double[]> inputData = new ArrayList<>();
	private static final List<Double[]> outputData = new ArrayList<>();
	private static final NeuralNetwork matrixNetwork = NeuralNetworkFactory.current()
			.createMatrixBasedNN()
			.addInputLayer(2)
			.addHiddenLayer(2)
			.addHiddenLayer(2)
			.addOutputLayer(1)
			.create();
	private static final NeuralNetwork objectNetwork = NeuralNetworkFactory.current()
			.createObjectBasedNN()
			.addInputLayer(2)
			.addHiddenLayer(10)
			.addHiddenLayer(10)
			.addHiddenLayer(10)
			.addHiddenLayer(10)
			.addOutputLayer(1)
			.create();
	private static final NeuralNetworkTrainer matrixTrainer = new NeuralNetworkTrainer();
	private static final NeuralNetworkTrainer objectTrainer = new NeuralNetworkTrainer();
	private static final SingleLayerTrainer singleLayerTrainer = new SingleLayerTrainer();
	private static final AtomicBoolean listeningToUserInput = new AtomicBoolean(false);
	private static final Thread listenerThread;

	static {
		inputData.add(new Double[] { 0.0, 0.0 });
		inputData.add(new Double[] { 0.0, 1.0 });
		inputData.add(new Double[] { 1.0, 0.0 });
		inputData.add(new Double[] { 1.0, 1.0 });

		outputData.add(new Double[] { 0.0 });
		outputData.add(new Double[] { 1.0 });
		outputData.add(new Double[] { 1.0 });
		outputData.add(new Double[] { 0.0 });

		listeningToUserInput.set(true);
		listenerThread = new Thread(Thread.currentThread().getThreadGroup(), () -> {
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			while (listeningToUserInput.get()) {
				try {
					br.readLine();
					matrixTrainer.stop();
					objectTrainer.stop();
					singleLayerTrainer.stop();
				} catch (IOException ignored) {}
			}
		});
		listenerThread.setName("TypingListenerThread");
		listenerThread.start();
		Thread.setDefaultUncaughtExceptionHandler((thread, exception) -> exception.printStackTrace(System.out));
	}

	public static void main(String[] args) {
		new Example().run();
	}

	private void printExpected() {
		System.out.println("Expected Results");
		for (int i = 0; i < inputData.size(); i++) {
			System.out.println(Arrays.toString(inputData.get(i)) + " => " + Arrays.toString(outputData.get(i)));
		}
		System.out.println();
	}

	private void printCalculated(NeuralNetwork neuralNetwork) {
		System.out.println("Current Results for the given Problem: ");
		for (Double[] input : inputData) {
			Double[] calculated = neuralNetwork.evaluate(input);
			long[] roundedCalculated = new long[calculated.length];
			for(int i = 0 ; i < calculated.length ; i++) {
				Double value = calculated[i];
				roundedCalculated[i] = Math.round(value);
			}
			System.out.println(Arrays.toString(input) + " => " + Arrays.toString(calculated) + "(" + Arrays.toString(roundedCalculated) + ")");
		}
		System.out.println();
	}

	void run() {
		testMatrixNetwork();
//		testObjectNetwork();
//		testSingleNeuron();
		listeningToUserInput.set(false);
	}

	private void testObjectNetwork() {
		objectTrainer.setTrainingInputData(inputData);
		objectTrainer.setTrainingOutputData(outputData);
		objectTrainer.setIndefiniteTrainingEnd();
		objectTrainer.setProbingTime(10000);
		objectTrainer.setMaximumError(0.0000000001);
		objectNetwork.setLearningRate(0.1);

		System.out.println("Starting Training of NN!");
		System.out.println("Maximum Trainings cycles: " + objectTrainer.getMaximumTrainingCycles());
		System.out.println("Maximum error: " + objectTrainer.getMaximumError());
		printExpected();
		System.out.println();
		printCalculated(objectNetwork);
		System.out.println("Start.");

		objectTrainer.train(objectNetwork);

		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println("Training complete!.");
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println("Training took: " + objectTrainer.getCurrentIteration() + " with an error of: " + objectTrainer.getCurrentCalculatedError());

		printExpected();
		System.out.println();
		printCalculated(objectNetwork);
	}

	private void testMatrixNetwork() {
		matrixTrainer.setTrainingInputData(inputData);
		matrixTrainer.setTrainingOutputData(outputData);
		matrixTrainer.setIndefiniteTrainingEnd();
		matrixTrainer.setProbingTime(10000);
		matrixTrainer.setMaximumError(0.00001);

		System.out.println("Starting Training of NN!");
		System.out.println("Maximum Trainings cycles: " + matrixTrainer.getMaximumTrainingCycles());
		System.out.println("Maximum error: " + matrixTrainer.getMaximumError());
		printExpected();
		System.out.println();
		printCalculated(matrixNetwork);
		System.out.println("Start.");

		matrixTrainer.train(matrixNetwork);

		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println("Training complete!.");
		System.out.println();
		System.out.println();
		System.out.println();
		System.out.println("Training took: " + matrixTrainer.getCurrentIteration() + " with an error of: " + matrixTrainer.getCurrentCalculatedError());

		printExpected();
		System.out.println();
		printCalculated(matrixNetwork);
	}

	private void testSingleNeuron() {
		Neuron orNeuron = new AbstractNeuron() {};

		Neuron inputOne = new SupplierNeuron();
		Neuron inputTwo = new SupplierNeuron();

		orNeuron.connectToInput(inputOne);
		orNeuron.connectToInput(inputTwo);

		final List<Double[]> inputData = new ArrayList<>();
		inputData.add(new Double[]{0.0 , 0.0});
		inputData.add(new Double[]{0.0 , 1.0});
		inputData.add(new Double[]{1.0 , 0.0});
		inputData.add(new Double[]{1.0 , 1.0});
		final List<Double[]> outputData = new ArrayList<>();
		outputData.add(new Double[]{0.0});
		outputData.add(new Double[]{1.0});
		outputData.add(new Double[]{1.0});
		outputData.add(new Double[]{1.0});

		singleLayerTrainer.train(Collections.singletonList(orNeuron), inputData, outputData);

		inputOne.setInputValue(0.0);
		inputOne.setInputValue(0.0);
		System.out.println("[0, 0] => " + orNeuron.guess());

		inputOne.setInputValue(0.0);
		inputOne.setInputValue(1.0);
		System.out.println("[0, 1] => " + orNeuron.guess());

		inputOne.setInputValue(1.0);
		inputOne.setInputValue(0.0);
		System.out.println("[1, 0] => " + orNeuron.guess());

		inputOne.setInputValue(1.0);
		inputOne.setInputValue(1.0);
		System.out.println("[1, 1] => " + orNeuron.guess());
	}
}
