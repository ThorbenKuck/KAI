package com.github.thorbenkuck.kai.neural;

import java.util.concurrent.ThreadLocalRandom;

public interface Connection {

	static  Connection create(Neuron source, Neuron target) {
		return new ConnectionImpl(source, target, randomWeight());
	}

	static Connection create(Neuron source, Neuron target, double weight) {
		return new ConnectionImpl(source, target, weight);
	}

	static  Connection create(Neuron source, Neuron target, double weight, double deltaWeight) {
		Connection connection = new ConnectionImpl(source, target, weight);
		connection.setDeltaWeight(deltaWeight);
		return connection;
	}

	static double randomWeight() {
		return ((ThreadLocalRandom.current().nextDouble() * 2) -1);
	}

	Neuron getSource();

	Neuron getTarget();

	double getWeight();

	double getDeltaWeight();

	void setDeltaWeight(double deltaWeight);

	void setWeight(double newWeight);
}
