package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.math.Matrix;
import com.github.thorbenkuck.kai.neural.ActivationFunction;
import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.NeuralNetwork;
import com.github.thorbenkuck.kai.neural.implementation.SigmoidActivationFunction;

import java.util.*;

class NeuralNetworkImpl implements NeuralNetwork {

	private final List<Matrix> weightMatrices = new ArrayList<>();
	private final List<Matrix> biasMatrices = new ArrayList<>();
	private double learningRate = 0.1;
	private ActivationFunction activationFunction = new SigmoidActivationFunction();

	NeuralNetworkImpl(int inputNeurons, List<Integer> hiddenLayersNeurons, int outputNeurons) {
		createMatrices(inputNeurons, hiddenLayersNeurons, outputNeurons);
	}

	private void createMatrices(int inputNeurons, List<Integer> hiddenLayersNeurons, int outputNeurons) {
		Queue<Integer> hiddenLayerQueue = new LinkedList<>(hiddenLayersNeurons);
		hiddenLayerQueue.add(outputNeurons);
		int current = inputNeurons;
		while (hiddenLayerQueue.peek() != null) {
			int temp = hiddenLayerQueue.poll();
			Matrix weights = new Matrix(temp, current);
			weights.randomize(- 1, 1);
			weightMatrices.add(weights);
			current = temp;
			Matrix bias = new Matrix(current, 1);
			bias.randomize(-1, 1);
			biasMatrices.add(bias);

		}
	}

	@Override
	public Double[] feedForward(final Double[] input) {
		Matrix outputMatrix = Matrix.fromArray(input);

		for (int i = 0; i < weightMatrices.size(); i++) {
			Matrix weightMatrix = weightMatrices.get(i);
			Matrix bias = biasMatrices.get(i);

			// Apply the weights, and the bias to the input.
			Matrix temp = Matrix.multiply(weightMatrix, outputMatrix);
			temp.plus(bias);
			temp.map(aDouble -> activationFunction.calculate(aDouble));

			// Save the current matrix to be used in the next cycle
			outputMatrix = temp;
		}

		return outputMatrix.to1DArray();
	}

	private List<Matrix> calculateMatricesStepByStep(Matrix input) {
		// redundant
		final List<Matrix> result = new ArrayList<>();
		result.add(input);
		Matrix outputMatrix = input;

		for (int i = 0; i < weightMatrices.size(); i++) {
			Matrix weightMatrix = weightMatrices.get(i);
			Matrix bias = biasMatrices.get(i);

			// Apply the weights, and the bias to the input.
			Matrix temp = Matrix.multiply(weightMatrix, outputMatrix);
			temp.plus(bias);
			temp.map(aDouble -> activationFunction.calculate(aDouble));

			// Save the current matrix to be used in the next cycle
			outputMatrix = temp;
			result.add(outputMatrix);
		}

		return result;
	}

	@Override
	public double train(Double[] inputs, Double[] expectedResults) {
		List<Matrix> allSteps = calculateMatricesStepByStep(Matrix.fromArray(inputs));
		Collections.reverse(allSteps);
		Queue<Matrix> outputQueue = new LinkedList<>(allSteps);
		Matrix targets = Matrix.fromArray(expectedResults);

		int pointer = weightMatrices.size() - 1;

		Matrix outputs = outputQueue.poll();
		Matrix next = outputQueue.poll();
		Matrix nextTranspose = Matrix.transpose(next);
		Matrix outputError = Matrix.subtract(targets, outputs);
		Matrix gradient = Matrix.map(outputs, aDouble -> activationFunction.calculateDerivative(aDouble));
		gradient.times(outputError);
		gradient.times(learningRate);

		Matrix weight_delta = Matrix.multiply(gradient, nextTranspose);
		Matrix currentWeights = weightMatrices.get(pointer);
		Matrix currentBias = biasMatrices.get(pointer);

		currentWeights.plus(weight_delta);
		currentBias.plus(gradient);

		Matrix lastWeights = currentWeights;
		outputs = next;
		--pointer;
		while(outputQueue.peek() != null) {
			currentWeights = weightMatrices.get(pointer);
			currentBias = biasMatrices.get(pointer);

			Matrix transposedLastWeights = Matrix.transpose(lastWeights);
			Matrix hiddenError = Matrix.multiply(transposedLastWeights, outputError);
			Matrix hiddenGradient = Matrix.map(outputs, aDouble -> activationFunction.calculateDerivative(aDouble));
			hiddenGradient.times(hiddenError);
			hiddenGradient.times(learningRate);

			next = Matrix.transpose(outputQueue.poll());
			weight_delta = Matrix.multiply(hiddenGradient, next);

			currentWeights.plus(weight_delta);
			currentBias.plus(hiddenGradient);

			--pointer;
		}


		return outputError.absoluteSum();
	}

	@Override
	public void reset() {
		for(Matrix matrix : weightMatrices) {
			matrix.randomize(-1, 1);
		}

		for(Matrix matrix : biasMatrices) {
			matrix.randomize(-1, 1);
		}
	}

	@Override
	public void setLearningRate(final double to) {
		this.learningRate = learningRate;
	}

	@Override
	public double getLearningRate() {
		return learningRate;
	}

	@Override
	public List<Matrix> matrixList() {
		return new ArrayList<>(weightMatrices);
	}

	@Override
	public String toString() {
		return "NeuralNetworkImpl{" + "weightMatrices=" + weightMatrices +
				'}';
	}

	// Package private for factory
	Collection<Layer> getAllLayer() {
		final List<Layer> layers = new ArrayList<>();
//		layers.add(inputLayer);
//		layers.addAll(hiddenLayers);
//		layers.add(outputLayer);

		return Collections.unmodifiableCollection(layers);
	}
}
