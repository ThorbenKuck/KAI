package com.github.thorbenkuck.kai.neural;

import com.github.thorbenkuck.kai.neural.implementation.SigmoidActivationFunction;

/**
 * Stateless activation functions do not need to be recreated all the time. Therefore, they should be created only once
 * (here).
 */
class ActivationFunctionCache {

	static ActivationFunction sigmoid = new SigmoidActivationFunction();

}
