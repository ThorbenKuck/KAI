package com.github.thorbenkuck.kai.neural;

import com.github.thorbenkuck.kai.math.Matrix;

import java.util.Collection;
import java.util.List;

public interface Layer extends Iterable<Neuron> {

	int getNumberOfNeurons();

	Neuron getNeuronAtIndex(int index);

	void setNeuron(int index, Neuron neuron);

	void randomizeInputConnectionWeights();

	void randomizeOutputConnectionWeights();

	Matrix guess();

	Matrix getOutputs();

	void setInputData(Double[] data);

	boolean isInput();

	boolean isOutput();

	boolean isHidden();

	Layer copy();

	Layer getNext();

	Layer getPrevious();

	void setNext(Layer next);

	void setPrevious(Layer previous);

	List<Neuron> copyAllNeurons();

	void resetError();
}
