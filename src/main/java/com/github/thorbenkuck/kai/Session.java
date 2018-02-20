package com.github.thorbenkuck.kai;

import com.github.thorbenkuck.kai.exceptions.TrainingFailedException;
import com.github.thorbenkuck.kai.neural.Layer;

import java.util.List;
import java.util.function.Supplier;

public interface Session {

	static Session create() {
		return new SessionImpl();
	}

	void createNeuralNetwork(Layer inputLayer, Layer hiddenLayer, Layer outputLayer);

	void createNeuralNetwork(Layer inputLayer, List<Layer> hiddenLayer, Layer outputLayer);

	void createNeuralNetwork(Layer inputLayer, Layer outputLayer);

	void setInputData(List<Double> inputData);

	void setInputDataSupplier(Supplier<List<Double>> inputDataSupplier);

	void setTrainingsData(List<Double[]> trainingsData);

	void setTrainingsDataSupplier(Supplier<List<Double[]>> trainingsDataSupplier);

	void trainFor(List<Double[]> outputData) throws TrainingFailedException;

	Double[] evaluate();
}
