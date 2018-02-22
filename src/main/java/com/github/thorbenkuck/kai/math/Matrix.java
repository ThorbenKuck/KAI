package com.github.thorbenkuck.kai.math;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

public class Matrix {

	private double[][] data;
	private int rows;
	private int columns;

	public Matrix(int size) {
		this(size, size);
	}

	public Matrix(int rows, int columns) {
		data = new double[rows][columns];
		this.rows = rows;
		this.columns = columns;
	}

	public Matrix(Matrix matrix) {
		this.data = matrix.data;
		this.rows = matrix.rows;
		this.columns = matrix.columns;
	}

	public static Matrix multiply(Matrix a, Matrix b) {
		if (a.columns != b.rows) {
			System.out.println(a);
			System.out.println();
			System.out.println(b);
			throw new RuntimeException("Illegal matrix dimensions.");
		}
		Matrix c = new Matrix(a.rows, b.columns);
		for (int i = 0; i < c.rows; i++) {
			for (int j = 0; j < c.columns; j++) {
				for (int k = 0; k < a.columns; k++) {
					c.data[i][j] += (a.data[i][k] * b.data[k][j]);
				}
			}
		}
		return c;
	}

	public static Matrix transpose(Matrix base) {
		int rows = base.rows;
		int columns = base.columns;
		Matrix a = new Matrix(columns, rows);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				a.data[j][i] = base.data[i][j];
			}
		}
		return a;
	}

	public static Matrix fromArray(Double[] array) {
		Matrix matrix = new Matrix(array.length, 1);
		for (int i = 0; i < array.length; i++) {
			matrix.setPoint(i, 0, array[i]);
		}
		return matrix;
	}

	public static Matrix subtract(Matrix a, Matrix b) {
		Matrix c = new Matrix(a);
		c.minus(b);

		return c;
	}

	public static Matrix map(Matrix a, Function<Double, Double> function) {
		Matrix result = new Matrix(a);
		result.map(function);

		return result;
	}

	public static Matrix map(Matrix a, MatrixFunction function) {
		Matrix result = new Matrix(a);
		result.map(function);

		return result;
	}

	private void requirePointInMatrix(int row, int column) {
		if (! pointInMatrix(row, column)) {
			throw new IllegalArgumentException("Point(" + row + "," + column + ") not in matrix");
		}
	}

	private void requireSameSize(Matrix a, Matrix b) {
		if (! sameSize(a, b)) {
			throw new RuntimeException("Illegal matrix dimensions.");
		}
	}

	public int columns() {
		return columns;
	}

	public int rows() {
		return rows;
	}

	public Matrix randomize(double upper) {
		return randomize(0, upper);
	}

	public Matrix randomize(double lower, double upper) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[i][j] = ThreadLocalRandom.current().nextDouble(lower, upper);
			}
		}
		return this;
	}

	public Matrix randomize() {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[i][j] = ThreadLocalRandom.current().nextInt(10);
			}
		}

		return this;
	}

	public void plus(Matrix b) {
		requireSameSize(b, this);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[i][j] += + b.data[i][j];
			}
		}
	}

	public void plus(double scalar) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[i][j] += scalar;
			}
		}
	}

	public void times(double scalar) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[i][j] *= scalar;
			}
		}
	}

	public void times(Matrix matrix) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[i][j] *= matrix.data[i][j];
			}
		}
	}

	public double sum() {
		double sum = 0.0;
		for(double[] array : data) {
			for(double currentData : array) {
				sum += currentData;
			}
		}

		return sum;
	}

	/**
	 * the Sum of all values, treated absolute
	 */
	public double absoluteSum() {
		double sum = 0.0;

		for(double[] array : data) {
			for(double currentData : array) {
				sum += Math.abs(currentData);
			}
		}

		return sum;
	}

	public void minus(Matrix b) {
		requireSameSize(b, this);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				data[i][j] -= b.data[i][j];
			}
		}
	}

	public void map(MatrixFunction function) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				double value = data[i][j];
				data[i][j] = function.apply(i, j, value);
			}
		}
	}

	public void map(Function<Double, Double> function) {
		map((i, j, d) -> function.apply(d));
	}

	public boolean equalTo(Matrix b) {
		if (! sameSize(this, b)) {
			return false;
		}
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				if (data[i][j] != b.data[i][j]) return false;
		return true;
	}

	public double getPoint(int row, int column) {
		if ((row < 0 || row >= rows) || column < 0 || column >= columns) {
			throw new IllegalArgumentException("Point(" + row + "," + column + ") not in matrix");
		}

		return data[row][column];
	}

	public void setPoint(int row, int column, double data) {
		requirePointInMatrix(row, column);
		this.data[row][column] = data;
	}

	public boolean pointInMatrix(int row, int column) {
		return ! (row < 0 || row >= rows || column < 0 || column >= columns);
	}

	public boolean sameSize(Matrix a, Matrix b) {
		return a.columns == b.columns && a.rows == b.rows;
	}

	public int getRows() {
		return rows;
	}

	public int getColumns() {
		return columns;
	}

	public void clear() {
		this.data = null;
		this.rows = 0;
		this.columns = 0;
	}

	public Double[][] toArray() {
		Double[][] toReturn = new Double[data.length][];
		for (int i = 0; i < data.length; i++) {
			double[] inner = data[i];
			toReturn[i] = new Double[inner.length];
			for (int j = 0; j < inner.length; j++) {
				double value = inner[j];
				toReturn[i][j] = value;
			}
		}

		return toReturn;
	}

	public Double[] to1DArray() {
		List<Double> doubleList = new ArrayList<>();

		for (double[] currentDataRow : data) {
			for (final double currentData : currentDataRow) {
				doubleList.add(currentData);
			}
		}

		return doubleList.toArray(new Double[doubleList.size()]);
	}

	@Override
	public String toString() {
		final StringBuilder stringBuilder = new StringBuilder();
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				stringBuilder.append(data[i][j]).append(" ");
			}
			stringBuilder.append(System.lineSeparator());
		}

		return stringBuilder.toString();
	}
}
