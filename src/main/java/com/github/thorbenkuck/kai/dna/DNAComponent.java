package com.github.thorbenkuck.kai.dna;

public abstract class DNAComponent <T> {

	private T state;

	public DNAComponent(T t) {
		this.state = t;
	}

	public T getState() {
		return state;
	}

	public void setState(T state) {
		this.state = state;
	}
}
