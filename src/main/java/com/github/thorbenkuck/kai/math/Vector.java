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

	public void add(Vector vector) {
		this.x += vector.x;
		this.y += vector.y;
		this.z += vector.z;
	}

	public void multiply(Vector vector) {
		this.x *= vector.x;
		this.y *= vector.y;
		this.z *= vector.z;
	}

	public void subtract(Vector vector) {
		this.x -= vector.x;
		this.y -= vector.y;
		this.z -= vector.z;
	}
}
