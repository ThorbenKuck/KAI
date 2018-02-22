package com.github.thorbenkuck.kai.neural.types;

import com.github.thorbenkuck.kai.neural.ActivationFunction;
import com.github.thorbenkuck.kai.neural.Connection;
import com.github.thorbenkuck.kai.neural.Neuron;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractNeuron implements Neuron {

	private final List<Connection> inputConnections = new ArrayList<>();
	private final List<Connection> outputConnections = new ArrayList<>();
	private final ActivationFunction activationFunction;
	private double inputValue;
	private double outputValue;
	private double error;

	public AbstractNeuron(ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	public AbstractNeuron() {
		this(ActivationFunction.sigmoid());
	}

	private double calculateInput() {
		if (inputConnections.isEmpty()) {
			return getInputValue();
		}

		double sum = 0.0;

		for (Connection connection : inputConnections) {
			Neuron inputNeuron = connection.getSource();
			sum += inputNeuron.getOutputValue() * connection.getWeight();
		}

		return sum;
	}

	private double correct(double expected, double guess) {
		return correct(expected, guess, 0.1);
	}

	private double correct(double expected, double guess, double learningRate) {
		double error = expected - guess;
		for (Connection connection : inputConnections) {
			double input = connection.getSource().getInputValue();
			connection.addToWeight(learningRate * error * input);
		}

		return Math.abs(error);
	}

	@Override
	public void calculate() {
		inputValue = calculateInput();
		outputValue = activationFunction.calculate(inputValue);
	}

	@Override
	public double guess() {
		calculate();
		return outputValue;
	}

	@Override
	public ActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public String toString() {
		return "AbstractNeuron{" + "activationFunction=" + activationFunction +
				", inputValue=" + inputValue +
				", outputValue=" + outputValue +
				'}';
	}

	protected Connection createInputConnection(Neuron target) {
		return Connection.create(target, this);
	}

	protected Connection createOutputConnection(Neuron target) {
		return Connection.create(this, target);
	}

	protected void addInputConnection(Connection connection) {
		inputConnections.add(connection);
	}

	protected void addOutputConnection(Connection connection) {
		outputConnections.add(connection);
	}

	@Override
	public double getInputValue() {
		return inputValue;
	}


	@Override
	public double getOutputValue() {
		return outputValue;
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
	public void setInputValue(double t) {
		this.inputValue = t;
	}


	/**
	 * Null means, the Connection is not to be found.
	 *
	 * @param searchFor
	 * @return
	 */
	@Override
	public Connection findConnectionTo(Neuron searchFor) {
		if (getAllInputConnections().isEmpty()) {
			System.out.println("Could not locate " + searchFor + " in " + this);
			return null;
		}

		for (Connection connection : getAllInputConnections()) {
			Neuron source = connection.getSource();
			if (source.equals(searchFor)) {
				return connection;
			}
			Connection toTest = source.findConnectionTo(searchFor);
			if (toTest != null) {
				return Connection.create(this, searchFor, connection.getWeight() + toTest.getWeight(), connection.getDeltaWeight() + toTest.getDeltaWeight());
			}
		}

		return null;
	}

	@Override
	public void connectToOutput(Neuron neuron) {
		addOutputConnection(createOutputConnection(neuron));
	}

	@Override
	public void connectToInput(Neuron neuron) {
		addInputConnection(createInputConnection(neuron));
	}

	@Override
	public double learn(Double[] inputs, double expected, double learningRate) {
		if (inputs.length != inputConnections.size()) {
			throw new IllegalArgumentException("Input data and inputValue Neurons do not match!");
		}
		for (int i = 0; i < inputs.length; i++) {
			Connection connection = inputConnections.get(i);
			connection.getSource().setInputValue(inputs[i]);
		}
		return learn(expected, learningRate);
	}

	@Override
	public double learn(double expected, double learningRate) {
		double guess = guess();
		if (guess == expected) {
			return 0.0;
		}
		return correct(expected, guess, learningRate);
	}

	@Override
	public boolean hasInputConnections() {
		return ! inputConnections.isEmpty();
	}

	@Override
	public boolean hasOutputConnection() {
		return ! outputConnections.isEmpty();
	}

	@Override
	public void setError(final double error) {
		this.error = error;
	}

	@Override
	public double getError() {
		return error;
	}

	@Override
	public boolean isInput() {
		return !hasInputConnections();
	}

	@Override
	public boolean isOutput() {
		return !hasOutputConnection();
	}
}
