package de.thorbenkuck.kai.neural;

import java.util.ArrayList;
import java.util.List;

public abstract class RandomCallNeuron<Result> extends Neuron<Result> {

	public RandomCallNeuron(float weight) {
		super(weight);
	}

	@Override
	public final void onCall() {
		getRandomNextNeuron().onImpulse(fire());
	}

	@Override
	public final void onImpulse(Result lastResult) {
		calc(lastResult);
		getRandomNextNeuron().onImpulse(fire());
	}

	private Neuron<Result> getRandomNextNeuron() {
		List<Neuron<Result>> toSort = new ArrayList<>(inputs);

		double totalWeight = 0.0d;
		for (Neuron<Result> i : toSort) {
			totalWeight += i.getWeight();
		}

		int randomIndex = -1;
		double random = Math.random() * totalWeight;
		for (int i = 0; i < toSort.size(); ++i)
		{
			random -= toSort.get(i).getWeight();
			if (random <= 0.0d)
			{
				randomIndex = i;
				break;
			}
		}
		return toSort.get(randomIndex);
	}
}
