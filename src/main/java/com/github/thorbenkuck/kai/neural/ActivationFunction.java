package com.github.thorbenkuck.kai.neural;

public interface ActivationFunction {

	double calculate(double t);

	double calculateDerivative(double t);

}
