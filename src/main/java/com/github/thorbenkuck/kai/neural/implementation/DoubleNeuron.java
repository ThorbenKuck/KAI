package com.github.thorbenkuck.kai.neural.implementation;

import com.github.thorbenkuck.kai.neural.AbstractNeuron;
import com.github.thorbenkuck.kai.neural.ActivationFunction;

public class DoubleNeuron extends AbstractNeuron {

	public DoubleNeuron() {
		super(aDouble -> {
			if(aDouble < 1) {
				return 0;
			} else {
				return 1;
			}
		});
	}
}
