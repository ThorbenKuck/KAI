package com.github.thorbenkuck.kai.neural.implementation;

import com.github.thorbenkuck.kai.neural.ActivationFunction;

public class SigmoidActivationFunction implements ActivationFunction {
	@Override
	public double calculate(final double t) {
		return 1 / (1 + Math.exp(- t));
	}

	@Override
	public double calculateDerivative(final double t) {
		double sigmoid = calculate(t);
		return sigmoid * (1 - sigmoid);
	}
}
