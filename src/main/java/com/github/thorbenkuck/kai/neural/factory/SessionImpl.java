package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.Session;
import com.github.thorbenkuck.kai.exceptions.TrainingFailedException;
import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.NeuralNetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

public class SessionImpl implements Session {

	private NeuralNetwork neuralNetwork;
	private Supplier<List<Double>> inputDataSupplier = ArrayList::new;
	private Supplier<List<Double[]>> trainingsDataSupplier = ArrayList::new;

	@Override
	public void setInputData(final List<Double> inputData) {
		setInputDataSupplier(() -> new ArrayList<>(inputData));
	}

	@Override
	public void setInputDataSupplier(Supplier<List<Double>> inputDataSupplier) {
		this.inputDataSupplier = inputDataSupplier;
	}

	@Override
	public void setTrainingsData(final List<Double[]> trainingsData) {
		setTrainingsDataSupplier(() -> new ArrayList<>(trainingsData));
	}

	@Override
	public void setTrainingsDataSupplier(Supplier<List<Double[]>> trainingsDataSupplier) {
		this.trainingsDataSupplier = trainingsDataSupplier;
	}

	@Override
	public void trainFor(List<Double[]> outputData) throws TrainingFailedException {
//		if(!neuralNetwork.train(trainingsDataSupplier.get(), outputData)) {
//			throw new TrainingFailedException();
//		}
	}

	@Override
	public Double[] evaluate() {
//		List<Double> result = neuralNetwork.calculate(inputDataSupplier.get());
//		return result.toArray(new Double[result.size()]);
		return new Double[0];
	}
}
