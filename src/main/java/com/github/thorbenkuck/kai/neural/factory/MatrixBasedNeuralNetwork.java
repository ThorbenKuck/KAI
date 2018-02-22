package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.math.Matrix;
import com.github.thorbenkuck.kai.neural.ActivationFunction;
import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.NeuralNetwork;
import com.github.thorbenkuck.kai.neural.implementation.SigmoidActivationFunction;

import java.util.*;

class MatrixBasedNeuralNetwork implements NeuralNetwork {

	private final List<Matrix> weightMatrices = new ArrayList<>();
	private final List<Matrix> biasMatrices = new ArrayList<>();
	private double learningRate = 0.1;
	private ActivationFunction activationFunction = new SigmoidActivationFunction();

	MatrixBasedNeuralNetwork(int inputNeurons, List<Integer> hiddenLayersNeurons, int outputNeurons) {
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

	private Double[] feedForward(final Double[] input) {
		List<Matrix> result = calculateMatricesStepByStep(Matrix.fromArray(input));
		Collections.reverse(result);
		Matrix outputMatrix = result.get(0);

		return outputMatrix.to1DArray();
	}

	@Override
	public Double[] evaluate(final Double[] input) {
		return feedForward(input);
	}

	private List<Matrix> calculateMatricesStepByStep(Matrix input) {
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
		final List<Matrix> allSteps = calculateMatricesStepByStep(Matrix.fromArray(inputs));
		Collections.reverse(allSteps);
		final Queue<Matrix> outputQueue = new LinkedList<>(allSteps);
		final Matrix targets = Matrix.fromArray(expectedResults);

		int pointer = weightMatrices.size() - 1;

		Matrix outputs = outputQueue.poll();
		Matrix next = outputQueue.poll();
		Matrix nextTranspose = Matrix.transpose(next);
		Matrix outputError = Matrix.subtract(targets, outputs);
		double sum = outputError.absoluteSum();
		Matrix gradient = Matrix.map(outputs, aDouble -> activationFunction.calculateDerivative(aDouble));
		gradient.times(outputError);
		gradient.times(learningRate);

		Matrix weight_delta = Matrix.multiply(gradient, nextTranspose);
		Matrix currentWeights = weightMatrices.get(pointer);
		Matrix currentBias = biasMatrices.get(pointer);

		currentWeights.plus(weight_delta);
		currentBias.plus(gradient);

		Matrix lastWeights = currentWeights;

		// CleanUp, to help with potential memory issues
		gradient.clear();
		nextTranspose.clear();

		outputs = next;
		--pointer;
		while(outputQueue.peek() != null) {
			currentWeights = new Matrix(weightMatrices.get(pointer));
			currentBias = new Matrix(biasMatrices.get(pointer));

			Matrix transposedLastWeights = Matrix.transpose(lastWeights);
			outputError = Matrix.multiply(outputError, transposedLastWeights);
			gradient = Matrix.map(outputs, aDouble -> activationFunction.calculateDerivative(aDouble));
			gradient.times(outputError);
			gradient.times(learningRate);

			next = Matrix.transpose(outputQueue.poll());
			weight_delta = Matrix.multiply(gradient, next);

			currentWeights.plus(weight_delta);
			currentBias.plus(gradient);
			lastWeights = currentWeights;

			// CleanUp, to help with potential memory issues
			transposedLastWeights.clear();

			--pointer;
		}

		outputError.clear();

		return sum;
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
		this.learningRate = to;
	}

	@Override
	public double getLearningRate() {
		return learningRate;
	}

	@Override
	public String toPrettyString() {
		return toString();
	}

	@Override
	public String toString() {
		return "MatrixBasedNeuralNetwork{" + "weightMatrices=" + weightMatrices +
				'}';
	}
}
