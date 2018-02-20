package com.github.thorbenkuck.kai.dna;

class DNAElement<T> {
	private T t;

	public DNAElement(T t) {
		this.t = t;
	}

	public T get() {
		return t;
	}

	public void set(T t) {
		this.t = t;
	}
}
