package com.github.thorbenkuck.kai.dna;

import java.util.Collections;
import java.util.Iterator;
import java.util.Queue;

public abstract class DNA<T extends DNAComponent> implements Iterable<T> {

	private DNAElement<T>[] components;
	private static final int defaultStartSize = 10;

	public DNA() {
		this(defaultStartSize);
	}

	public DNA(int size) {
		components = new DNAElement[size];
	}

	public T getAt(int index) {
		return components[index].get();
	}

	public void setAt(int index, T dnaComponent) {
		if(index < 0 || index > components.length) {
			throw new IllegalArgumentException("Trying to current a DNAComponent outside of the DNA");
		}
		if(components[index] == null) {
			components[index] = new DNAElement<>(dnaComponent);
		} else {
			components[index].set(dnaComponent);
		}
	}

	abstract void mutate();

	@Override
	public Iterator<T> iterator() {
		return new DNAIterator(components);
	}

	private class DNAIterator implements Iterator<T> {

		Queue<DNAElement<T>> workQueue;

		DNAIterator(DNAElement<T>[] components) {
			Collections.addAll(workQueue, components);
		}

		@Override
		public boolean hasNext() {
			return workQueue.peek() != null;
		}

		@Override
		public T next() {
			return workQueue.remove().get();
		}
	}
}