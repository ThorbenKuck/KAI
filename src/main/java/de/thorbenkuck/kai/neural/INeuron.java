package de.thorbenkuck.kai.neural;

public interface INeuron<Result> {
	void connect(Neuron<Result>... neurons);
	void chain(Neuron<Result>... neurons);
	Result calc();
	Result calc(Result previousResult);
	void onCall();
	void onImpulse(Result lastResult);
}
