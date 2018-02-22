package com.github.thorbenkuck.kai.neural.factory;

public interface NeuralNetworkFactory {

	static NeuralNetworkFactory current() {
		return NNFCache.nnf;
	}

	MatrixBasedNeuralNetworkFactory createMatrixBasedNN();

	ObjectBasedNeuralNetworkFactory createObjectBasedNN();
}
