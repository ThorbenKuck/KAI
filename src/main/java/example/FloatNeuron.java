package example;

import de.thorbenkuck.kai.neural.Neuron;

public abstract class FloatNeuron extends Neuron<Float> {
	public FloatNeuron(float weight) {
		super(weight);
	}

	@Override
	public void onCall() {
	}

	@Override
	public void onImpulse(Float lastResult) {
	}

	@Override
	public Float fire(Float lastInput) {
		return weight;
	}
}
