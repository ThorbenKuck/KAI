package com.github.thorbenkuck.kai.math;

public class Vector {

	private double x;
	private double y;
	private double z;

	public Vector(double x, double y, double z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}

	public Vector(double x, double y) {
		this(x, y, 0);
	}

	public Vector() {
		this(0, 0);
	}

	public Vector(Vector vector) {
		this(vector.x, vector.y, vector.z);
	}

	public void addBy(Vector vector) {
		this.x += vector.x;
		this.y += vector.y;
		this.z += vector.z;
	}

	public void multiplyBy(Vector vector) {
		this.x *= vector.x;
		this.y *= vector.y;
		this.z *= vector.z;
	}

	public void multiplyBy(int times) {
		multiplyBy(new Vector(times, times, times));
	}

	public void multiplyBy(double times) {
		multiplyBy(new Vector(times, times, times));
	}

	public void subtractBy(Vector vector) {
		this.x -= vector.x;
		this.y -= vector.y;
		this.z -= vector.z;
	}

	@Override
	public String toString() {
		final StringBuilder stringBuilder = new StringBuilder();
		final String ls = System.lineSeparator();
		stringBuilder.append(x).append(ls).append(y).append(ls).append(z).append(ls);
		return stringBuilder.toString();
	}
}
