package example;

import de.thorbenkuck.kai.neural.Neuron;

public class Example {
	public static void main(String[] args) {
		FloatNeuron plus = new FloatNeuron(1.0f) {
			@Override
			public Float fire() {
				float result = 0;
				for(Neuron<Float> neuron : inputs) {
					result += neuron.fire();
				}
				return result;
			}
		};

		FloatNeuron one = new FloatNeuron(1.0f) {
			@Override
			public Float fire() {
				return weight;
			}
		};

		FloatNeuron two = new FloatNeuron(2.0f) {
			@Override
			public Float fire() {
				return weight;
			}
		};

		plus.connect(one, two);
		plus.calc();
		System.out.println(plus.getCachedResult());

		BooleanNeuron xor = new BooleanNeuron(1.0f) {
			@Override
			public Boolean fire() {
				boolean toReturn = false;
				for(Neuron<Boolean> bool : inputs) {
					toReturn ^= bool.fire();

				}
				return toReturn;
			}
		};

		BooleanNeuron not = new BooleanNeuron(1.0f) {
			@Override
			public Boolean fire() {
				final boolean[] toReturn = new boolean[1];
				inputs.stream().findFirst().ifPresent(neuron -> toReturn[0] = neuron.fire());
				return !toReturn[0];
			}
		};

		BooleanNeuron trueOne = new BooleanNeuron(2.0f) {
			@Override
			public Boolean fire() {
				return false;
			}
		};

		BooleanNeuron trueOneTwo = new BooleanNeuron(2.0f) {
			@Override
			public Boolean fire() {
				return true;
			}
		};

		BooleanNeuron falseOne = new BooleanNeuron(2.0f) {
			@Override
			public Boolean fire() {
				return false;
			}
		};

		xor.connect(trueOne, falseOne, trueOneTwo);
		not.connect(xor);
		xor.calc();
		System.out.println(xor.getCachedResult());
		not.calc();
		System.out.println(not.getCachedResult());
	}
}
