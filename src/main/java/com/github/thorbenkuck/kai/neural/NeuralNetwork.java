package com.github.thorbenkuck.kai.neural;

import com.github.thorbenkuck.kai.math.Matrix;

import java.util.List;

public interface NeuralNetwork {

	Double[] feedForward(Double[] input);

	double train(Double[] inputs, Double[] answer);

	void reset();

	void setLearningRate(double to);

	double getLearningRate();

	List<Matrix> matrixList();

}
