package com.github.thorbenkuck.kai.neural.factory;

import com.github.thorbenkuck.kai.math.Matrix;
import com.github.thorbenkuck.kai.neural.Layer;
import com.github.thorbenkuck.kai.neural.Neuron;

import java.util.*;

class LayerImpl implements Layer {

	private final List<Neuron> neurons;

	public LayerImpl(List<Neuron> neuronList) {
		neurons = Collections.unmodifiableList(neuronList);
	}

	/**
	 * Returns an iterator over elements of type {@code T}.
	 *
	 * @return an Iterator.
	 */
	@Override
	public Iterator<Neuron> iterator() {
		return new QueuedIterator(neurons);
	}

	private class QueuedIterator implements Iterator<Neuron> {

		private final Queue<Neuron> core;
		private Neuron current;

		private QueuedIterator(final Collection<Neuron> collection) {
			this.core = new LinkedList<>(collection);
		}

		/**
		 * Returns {@code true} if the iteration has more elements.
		 * (In other words, returns {@code true} if {@link #next} would
		 * return an element rather than throwing an exception.)
		 *
		 * @return {@code true} if the iteration has more elements
		 */
		@Override
		public boolean hasNext() {
			return core.peek() != null;
		}

		/**
		 * Returns the next element in the iteration.
		 *
		 * @return the next element in the iteration
		 * @throws NoSuchElementException if the iteration has no more elements
		 */
		@Override
		public Neuron next() {
			if (! hasNext()) {
				throw new NoSuchElementException();
			}
			current = core.poll();
			return current;
		}

		/**
		 * Removes from the underlying collection the last element returned
		 * by this iterator (optional operation).  This method can be called
		 * only once per call to {@link #next}.  The behavior of an iterator
		 * is unspecified if the underlying collection is modified while the
		 * iteration is in progress in any way other than by calling this
		 * method.
		 *
		 * @throws UnsupportedOperationException if the {@code remove}
		 *                                       operation is not supported by this iterator
		 * @throws IllegalStateException         if the {@code next} method has not
		 *                                       yet been called, or the {@code remove} method has already
		 *                                       been called after the last call to the {@code next}
		 *                                       method
		 * @implSpec The default implementation throws an instance of
		 * {@link UnsupportedOperationException} and performs no other action.
		 */
		@Override
		public void remove() {
			neurons.remove(current);
		}
	}
}
