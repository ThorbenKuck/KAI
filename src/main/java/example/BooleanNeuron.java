package example;

import de.thorbenkuck.kai.neural.Neuron;

public abstract class BooleanNeuron extends Neuron<Boolean> {
	public BooleanNeuron(float weight) {
		super(weight);
	}

	@Override
	public void onCall() {
	}

	@Override
	public void onImpulse(Boolean lastResult) {
	}

	@Override
	public Boolean fire(Boolean lastInput) {
		return weight < 0;
	}
}
