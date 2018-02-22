package com.github.thorbenkuck.kai.datatypes;

import java.util.*;

public class QueuedIterator<T> implements Iterator<T> {

	private final Queue<T> core;
	private final Collection<T> original;
	private T current;

	public QueuedIterator(final Collection<T> collection) {
		this.core = new LinkedList<>(collection);
		this.original = collection;
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
	public T next() {
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
		original.remove(current);
	}

}
