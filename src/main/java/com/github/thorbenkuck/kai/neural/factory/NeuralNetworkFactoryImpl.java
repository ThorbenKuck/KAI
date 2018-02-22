package com.github.thorbenkuck.kai.neural.factory;

class NeuralNetworkFactoryImpl implements NeuralNetworkFactory {
	@Override
	public MatrixBasedNeuralNetworkFactory createMatrixBasedNN() {
		return new MatrixBasedNeuralNetworkFactoryImpl();
	}

	@Override
	public ObjectBasedNeuralNetworkFactory createObjectBasedNN() {
		return new ObjectBasedNeuralNetworkFactoryImpl();
	}
}
