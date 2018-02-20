package com.github.thorbenkuck.kai;

import com.github.thorbenkuck.kai.exceptions.TrainingFailedException;
import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.implementation.DoubleNeuron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Example {
	public static void main(String[] args) throws TrainingFailedException {
		final Layer inputLayer = new Layer(new DoubleNeuron(), new DoubleNeuron());
		final Layer hiddenLayer = new Layer(new DoubleNeuron(), new DoubleNeuron());
		final Layer outputLayer = new Layer(new DoubleNeuron());

		Session session = Session.create();
		session.createNeuralNetwork(inputLayer, hiddenLayer, outputLayer);

		List<Double[]> inputData = new ArrayList<>();
		inputData.add(new Double[]{0.0, 0.0});
		inputData.add(new Double[]{1.0, 0.0});
		inputData.add(new Double[]{0.0, 1.0});
		inputData.add(new Double[]{1.0, 1.0});

		List<Double[]> outputData = new ArrayList<>();
		outputData.add(new Double[]{0.0});
		outputData.add(new Double[]{1.0});
		outputData.add(new Double[]{1.0});
		outputData.add(new Double[]{0.0});

		session.setTrainingsData(inputData);

		System.out.println("Training data accumulated!");
		System.out.println("Input Data: " + inputData);
		System.out.println("Output Data: " + outputData);
		System.out.println();
		System.out.println("Training neural net..");
		session.trainFor(outputData);
		System.out.println("Training successful");
		Double[] input1 = {1.0, 1.0};
		Double[] input2 = {1.0, 0.0};
		System.out.print("Calculating 1 xor 1: ");
		session.setInputData(Arrays.asList(input1));
		System.out.println(Arrays.toString(session.evaluate()));
		System.out.print("Calculating 1 xor 0: ");
		session.setInputData(Arrays.asList(input2));
		System.out.println(Arrays.toString(session.evaluate()));
	}
}
