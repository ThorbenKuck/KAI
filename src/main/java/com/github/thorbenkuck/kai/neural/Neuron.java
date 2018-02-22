package com.github.thorbenkuck.kai.neural;

import java.util.List;

public interface Neuron {

	void calculate();

	double guess();

	ActivationFunction getActivationFunction();

	void setInputValue(double input);

	double getInputValue();

	double getOutputValue();

	List<Connection> getAllInputConnections();

	List<Connection> getAllOutputConnections();

	Connection findConnectionTo(Neuron neuron);

	void connectToOutput(Neuron neuron);

	void connectToInput(Neuron neuron);

	double learn(Double[] inputs, double expected, double learningRate);

	double learn(double expected, double learningRate);

	boolean hasInputConnections();

	boolean hasOutputConnection();

	boolean isOutput();

	boolean isInput();

	void setError(double error);

	double getError();
}
