package com.github.thorbenkuck.kai.neural;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractNeuron implements Neuron {

	private final List<Connection> inputConnections = new ArrayList<>();
	private final List<Connection> outputConnections = new ArrayList<>();
	private final ActivationFunction activationFunction;
	private double input;
	private double output;

	public AbstractNeuron(ActivationFunction activationFunction) {
		this.activationFunction = activationFunction;
	}

	@Override
	public void calculate() {
		input = calculateInput();
		output = activationFunction.activate(input);
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
	public Connection getConnectionTo(Neuron searchFor) {
		if(getAllInputConnections().isEmpty()) {
			System.out.println("Could not locate " + searchFor + " in " + this);
			return null;
		}

		for(Connection connection : getAllInputConnections()) {
			Neuron source = connection.getSource();
			if(source.equals(searchFor)) {
				return connection;
			}
			Connection toTest = source.getConnectionTo(searchFor);
			if(toTest != null) {
				return Connection.create(this, searchFor, connection.getWeight() + toTest.getWeight(), connection.getDeltaWeight() + toTest.getDeltaWeight());
			}
		}

		return null;
	}

	private double calculateInput() {
		if(inputConnections.isEmpty()) {
			return getInput();
		}

		double input = 0.0;

		for(Connection connection : inputConnections) {
			Neuron toCalculate = connection.getSource();
			toCalculate.calculate();
			double sum = toCalculate.getOutput() + input;
			input = sum * connection.getWeight();
		}

		return input;
	}

	@Override
	public void connectToOutput(Neuron neuron) {
		outputConnections.add(Connection.create(this, neuron));
	}

	@Override
	public void connectToInput(Neuron neuron) {
		inputConnections.add(Connection.create(neuron, this));
	}
}
