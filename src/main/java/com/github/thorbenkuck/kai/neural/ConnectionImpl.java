package com.github.thorbenkuck.kai.neural;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

class ConnectionImpl implements Connection {

	private final Neuron source;
	private final Neuron target;
	private double weight;
	private double deltaWeight = 0.0;
	private final AtomicBoolean weightsFixed = new AtomicBoolean(false);

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
		if(weightsFixed.get()) {
			return;
		}
		this.deltaWeight = deltaWeight;
	}

	@Override
	public void setWeight(double weight) {
		if(weightsFixed.get()) {
			return;
		}
		synchronized (this) {
			this.weight = weight;
		}
	}

	@Override
	public void addToWeight(double amount) {
		if(weightsFixed.get()){
			return;
		}
		synchronized (this) {
			this.weight += amount;
		}
	}

	@Override
	public void fixWeight() {
		weightsFixed.set(true);
	}

	@Override
	public String toString() {
		return source + " <-" + weight + "-> " + target;
	}
}
