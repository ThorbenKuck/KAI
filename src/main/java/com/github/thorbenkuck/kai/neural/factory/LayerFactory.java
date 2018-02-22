package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.Neuron;
import com.github.thorbenkuck.kai.neural.types.DefaultNeuron;
import com.github.thorbenkuck.kai.neural.types.SupplierNeuron;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

class LayerFactory implements Function<Integer, Layer> {
	private List<Neuron> createNeurons(int howMany) {
		final List<Neuron> neurons = new ArrayList<>();
		for (int i = 0; i < howMany; i++) {
			neurons.add(new DefaultNeuron());
		}
		return neurons;
	}

	/**
	 * Applies this function to the given argument.
	 *
	 * @param integer the function argument
	 * @return the function result
	 */
	@Override
	public LayerImpl apply(final Integer integer) {
		List<Neuron> neurons = createNeurons(integer);
		return new LayerImpl(neurons);
	}

	public LayerImpl createSupplierLayer(final int numberNeurons) {
		final List<Neuron> neurons = new ArrayList<>();
		for (int i = 0; i < numberNeurons; i++) {
			neurons.add(new SupplierNeuron());
		}

		return new LayerImpl(neurons);
	}
}
