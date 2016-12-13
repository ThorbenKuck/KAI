package de.thorbenkuck.kai.neural;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class Neuron<Result> implements INeuron<Result> {

	protected List<Neuron<Result>> inputs = new ArrayList<>();
	protected List<Neuron<Result>> outputs = new ArrayList<>();
	protected Result lastInput;
	protected float weight;
	private Result cachedResult;

	public Neuron(float weight) {
		this.weight = weight;
	}

	@Override
	public final void connect(Neuron<Result>... neurons) {
		Collections.addAll(inputs, neurons);
	}

	@Override
	public final void chain(Neuron<Result>... neurons) {
		Collections.addAll(outputs, neurons);
	}

	@Override
	public final Result calc() {
		cacheResult();
		return cachedResult;
	}

	@Override
	public final Result calc(Result previousResult) {
		lastInput = previousResult;
		cacheResult(fire(lastInput));
		return cachedResult;
	}

	/**
	 * What to do, when something calls this neuron
	 */
	@Override
	public abstract void onCall();

	/**
	 * What to do, when this should call other neurons
	 */
	@Override
	public abstract void onImpulse(Result lastResult);

	public final float getWeight() {
		return weight;
	}

	public final void setWeight(float weight) {
		this.weight = weight;
		AtomicInteger integer;
	}

	public Result getCachedResult() {
		return cachedResult;
	}

	protected void cacheResult(Result result) {
		cachedResult = result;
	}

	public void cacheResult() {
		cacheResult(fire());
	}

	public abstract Result fire();

	public abstract Result fire(Result lastInput);
}
