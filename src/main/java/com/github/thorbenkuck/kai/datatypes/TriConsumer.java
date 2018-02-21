package com.github.thorbenkuck.kai.datatypes;

@FunctionalInterface
public interface TriConsumer<T, U, V> {

	void consume(T t, U u, V v);

}
