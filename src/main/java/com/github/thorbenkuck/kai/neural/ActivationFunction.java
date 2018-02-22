package com.github.thorbenkuck.kai.neural;

public interface ActivationFunction {

	static ActivationFunction sigmoid() {
		return ActivationFunctionCache.sigmoid;
	}

	double calculate(double t);

	double calculateDerivative(double t);

}
