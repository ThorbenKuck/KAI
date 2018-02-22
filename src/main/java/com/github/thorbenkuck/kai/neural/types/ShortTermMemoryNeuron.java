package com.github.thorbenkuck.kai.neural.types;

import com.github.thorbenkuck.kai.neural.ActivationFunction;
import com.github.thorbenkuck.kai.neural.Connection;

public class ShortTermMemoryNeuron extends AbstractNeuron {

	public ShortTermMemoryNeuron() {
		super();
		selfConnect();
	}

	public ShortTermMemoryNeuron(ActivationFunction activationFunction) {
		super(activationFunction);
		selfConnect();
	}

	private void selfConnect() {
		Connection connection = createInputConnection(this);
		connection.setWeight(1.0);
		connection.setDeltaWeight(1.0);
		connection.fixWeight();
		addInputConnection(connection);
	}

	@Override
	public String toString() {
		return "ShortTermMemoryNeuron{" + "activationFunction=" + getActivationFunction() +
				", input=" + getInputValue() +
				", output=" + getOutputValue() +
				'}';
	}
}
