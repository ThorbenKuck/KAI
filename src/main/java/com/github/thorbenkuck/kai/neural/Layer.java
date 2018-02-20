package com.github.thorbenkuck.kai.neural;

import java.util.*;

public class Layer implements Iterable<Neuron> {

	private final List<Neuron> neuronList = new ArrayList<>();

	public Layer() {
	}

	public Layer(Neuron... neurons) {
		neuronList.addAll(Arrays.asList(neurons));
	}

	public int size() {
		return neuronList.size();
	}

	public Neuron get(int index) {
		return neuronList.get(index);
	}

	@Override
	public Iterator<Neuron> iterator() {
		return new LayerIterator(neuronList);
	}

	public void calculate() {
		for(Neuron neuron : neuronList) {
			neuron.calculate();
		}
	}

	private class LayerIterator implements Iterator<Neuron> {

		private final Queue<Neuron> leftoverElements;

		private LayerIterator(Collection<Neuron> collection) {
			this.leftoverElements = new LinkedList<>(collection);
		}

		@Override
		public boolean hasNext() {
			return leftoverElements.peek() != null;
		}

		@Override
		public Neuron next() {
			return leftoverElements.poll();
		}
	}
}
