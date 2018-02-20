package com.github.thorbenkuck.kai.neural;

import java.util.List;

public interface Neuron {

	void calculate();

	ActivationFunction getActivationFunction();

	void setInput(double input);

	double getInput();

	double getOutput();

	List<Connection> getAllInputConnections();

	List<Connection> getAllOutputConnections();

	Connection getConnectionTo(Neuron neuron);

	void connectToOutput(Neuron neuron);

	void connectToInput(Neuron neuron);
}
