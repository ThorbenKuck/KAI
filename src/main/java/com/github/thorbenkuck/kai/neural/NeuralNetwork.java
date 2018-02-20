package com.github.thorbenkuck.kai.neural;

import java.util.ArrayList;
import java.util.List;

public interface NeuralNetwork {

	static NeuralNetwork create(Layer inputLayer, Layer outputLayer) {
		return create(inputLayer, new ArrayList<>(), outputLayer);
	}

	static NeuralNetwork create(Layer inputLayer, List<Layer> hiddenLayer, Layer outputLayer) {
		return new NeuralNetworkImpl(inputLayer, hiddenLayer, outputLayer);
	}

	boolean train(List<Double[]> inputData, List<Double[]> outputData);

	List<Double> calculate(Double[] inputData);

	List<Double> calculate(List<Double> inputData);

	void setMaxIterations(int maxIterations);
}
