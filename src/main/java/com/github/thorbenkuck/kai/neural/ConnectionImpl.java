package com.github.thorbenkuck.kai.neural;

import java.util.Objects;

class ConnectionImpl implements Connection {

	private final Neuron source;
	private final Neuron target;
	private double weight;
	private double deltaWeight = 0.0;

	ConnectionImpl(Neuron source, Neuron target, double weight) {
		Objects.requireNonNull(source);
		Objects.requireNonNull(target);
		this.source = source;
		this.target = target;
		this.weight = weight;
	}

	@Override
	public Neuron getSource() {
		return source;
	}

	@Override
	public Neuron getTarget() {
		return target;
	}

	@Override
	public double getWeight() {
		synchronized (this) {
			return weight;
		}
	}

	@Override
	public double getDeltaWeight() {
		return deltaWeight;
	}

	@Override
	public void setDeltaWeight(double deltaWeight) {
		this.deltaWeight = deltaWeight;
	}

	@Override
	public void setWeight(double weight) {
		synchronized (this) {
			this.weight = weight;
		}
	}
}
