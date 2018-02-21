package com.github.thorbenkuck.kai;

import com.github.thorbenkuck.kai.exceptions.TrainingFailedException;

import java.util.List;
import java.util.function.Supplier;

public interface Session {

	void setInputData(List<Double> inputData);

	void setInputDataSupplier(Supplier<List<Double>> inputDataSupplier);

	void setTrainingsData(List<Double[]> trainingsData);

	void setTrainingsDataSupplier(Supplier<List<Double[]>> trainingsDataSupplier);

	void trainFor(List<Double[]> outputData) throws TrainingFailedException;

	Double[] evaluate();
}
