package com.github.thorbenkuck.kai.neural;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractPerceptron implements Neuron {

	private final List<Connection> inputConnections = new ArrayList<>();
	private final List<Connection> outputConnections = new ArrayList<>();
	private final ActivationFunction activationFunction;
	private double input;
	private double output;

	public AbstractPerceptron(ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	private double calculateInput() {
		if(inputConnections.isEmpty()) {
			return getInput();
		}

		double sum = 0.0;

		for(Connection connection : inputConnections) {
			Neuron inputNeuron = connection.getSource();
			sum += inputNeuron.getOutput() * connection.getWeight();
		}

		return sum;
	}

	@Override
	public void calculate() {
		input = calculateInput();
		output = activationFunction.calculate(input);
	}

	@Override
	public double guess() {
		calculate();
		return output;
	}

	@Override
	public double getInput() {
		return input;
	}

	@Override
	public double getOutput() {
		return output;
	}

	@Override
	public List<Connection> getAllInputConnections() {
		return inputConnections;
	}

	@Override
	public List<Connection> getAllOutputConnections() {
		return outputConnections;
	}

	@Override
	public void setInput(double t) {
		this.input = t;
	}

	@Override
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	/**
	 * Null means, the Connection is not to be found.
	 * @param searchFor
	 * @return
	 */
	@Override
	public Connection findConnectionTo(Neuron searchFor) {
		if(getAllInputConnections().isEmpty()) {
			System.out.println("Could not locate " + searchFor + " in " + this);
			return null;
		}

		for(Connection connection : getAllInputConnections()) {
			Neuron source = connection.getSource();
			if(source.equals(searchFor)) {
				return connection;
			}
			Connection toTest = source.findConnectionTo(searchFor);
			if(toTest != null) {
				return Connection.create(this, searchFor, connection.getWeight() + toTest.getWeight(), connection.getDeltaWeight() + toTest.getDeltaWeight());
			}
		}

		return null;
	}

	@Override
	public void connectToOutput(Neuron neuron) {
		outputConnections.add(Connection.create(this, neuron));
	}

	@Override
	public void connectToInput(Neuron neuron) {
		inputConnections.add(Connection.create(neuron, this));
	}

	@Override
	public double learn(Double[] inputs, double expected, double learningRate) {
		for(int i = 0 ; i < inputs.length ; i++) {
			Connection connection = inputConnections.get(i);
			connection.getSource().setInput(inputs[i]);
		}
		calculate();
		if(output == expected) {
			return 0.0;
		}
		return correct(expected, learningRate);
	}

	@Override
	public double correct(double expected) {
		return correct(expected, 0.5);
	}

	@Override
	public double correct(double expected, double learningRate) {
		double guess = output;
		double error = expected - guess;
		for(Connection connection : inputConnections) {
			double input = connection.getSource().getInput();
			connection.addToWeight(error * input * learningRate);
			System.out.println("new weight: " + connection.getWeight());
		}

		return error;
	}
}
