package com.github.thorbenkuck.kai.neural.types;

public class SupplierNeuron extends AbstractNeuron {

	@Override
	public double getOutputValue() {
		return getInputValue();
	}

	@Override
	public double guess() {
		return getInputValue();
	}

	@Override
	public String toString() {
		return "SupplierNeuron{" + "value=" + getOutputValue() + '}';
	}

}
