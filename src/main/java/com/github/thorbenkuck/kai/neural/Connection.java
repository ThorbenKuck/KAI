package com.github.thorbenkuck.kai.neural;

import java.util.concurrent.ThreadLocalRandom;

public interface Connection {

	static  Connection create(Neuron source, Neuron target) {
		return create(source, target, randomWeight());
	}

	static Connection create(Neuron source, Neuron target, double weight) {
		return create(source, target, weight, randomWeight());
	}

	static  Connection create(Neuron source, Neuron target, double weight, double deltaWeight) {
		Connection connection = new ConnectionImpl(source, target, weight);
		connection.setDeltaWeight(deltaWeight);
		return connection;
	}

	static double randomWeight() {
		return ThreadLocalRandom.current().nextDouble(-1, 1);
	}

	Neuron getSource();

	Neuron getTarget();

	double getWeight();

	double getDeltaWeight();

	void setDeltaWeight(double deltaWeight);

	void setWeight(double newWeight);

	void addToWeight(double amount);

	void fixWeight();
}
