package com.github.thorbenkuck.kai.neural;

import java.util.List;

public interface Neuron {

	void calculate();

	double guess();

	ActivationFunction getActivationFunction();

	void setInput(double input);

	double getInput();

	double getOutput();

	List<Connection> getAllInputConnections();

	List<Connection> getAllOutputConnections();

	Connection findConnectionTo(Neuron neuron);

	void connectToOutput(Neuron neuron);

	void connectToInput(Neuron neuron);

	double learn(Double[] inputs, double expected, double learningRate);

	double correct(double expected);

	double correct(double expected, double learningRate);
}
