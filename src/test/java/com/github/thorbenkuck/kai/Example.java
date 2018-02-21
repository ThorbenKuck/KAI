package com.github.thorbenkuck.kai;

import com.github.thorbenkuck.kai.neural.NeuralNetwork;
import com.github.thorbenkuck.kai.neural.NeuralNetworkTrainer;
import com.github.thorbenkuck.kai.neural.factory.NeuralNetworkFactory;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.SceneAntialiasing;
import javafx.scene.canvas.Canvas;
import javafx.scene.image.PixelWriter;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiConsumer;

//public class Example extends Application {
public class Example {

	private static final List<Double[]> inputData = new ArrayList<>();
	private static final List<Double[]> outputData = new ArrayList<>();
	private static final NeuralNetwork neuralNetwork = NeuralNetworkFactory.current()
			.addInputLayer(2)
			.addHiddenLayer(2)
			.addOutputLayer(1)
			.create();
	private static final NeuralNetworkTrainer trainer = new NeuralNetworkTrainer();
//	private int windowWidth = 800;
//	private int windowHeight = 800;
//	private int pixelDimensions = 400;
//	private Rectangle[][] pixels;

	static {
		inputData.add(new Double[] { 0.0, 0.0 });
		inputData.add(new Double[] { 0.0, 1.0 });
		inputData.add(new Double[] { 1.0, 0.0 });
		inputData.add(new Double[] { 1.0, 1.0 });

		outputData.add(new Double[] { 0.0 });
		outputData.add(new Double[] { 1.0 });
		outputData.add(new Double[] { 1.0 });
		outputData.add(new Double[] { 0.0 });

		neuralNetwork.setLearningRate(0.1);

		Thread other = new Thread(Thread.currentThread().getThreadGroup(), () -> {
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			String entered = "";
			while (! entered.equals("stopNeuralNetwork")) {
				try {
					br.readLine();
					entered = "stopNeuralNetwork";
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			trainer.stop();
		});
		other.setName("TypingListenerThread");
		other.start();
		Thread.setDefaultUncaughtExceptionHandler((thread, exception) -> {
			exception.printStackTrace(System.out);
		});
	}

	private void printExpected() {
		System.out.println("Expected Results");
		for (int i = 0; i < inputData.size(); i++) {
			System.out.println(Arrays.toString(inputData.get(i)) + " => " + Arrays.toString(outputData.get(i)));
		}
		System.out.println();
	}

//	private synchronized void applyPixels() {
//		int width = pixels.length;
//		for (int x = 0; x < width ; x++) {
//			int height = pixels[x].length;
//			for (int y = 0; y < height ; y++) {
//				Double xValue = ((x / (double) width) * 2) - 1;
//				Double yValue = ((y / (double) height) * 2) - 1;
//
//				Double result = neuralNetwork.feedForward(new Double[] { xValue, yValue })[0];
//				Rectangle rectangle = pixels[x][y];
//				rectangle.setFill(Color.gray(result));
//			}
//		}
//	}

	void run() {
		trainer.setTrainingInputData(inputData);
		trainer.setTrainingOutputData(outputData);
		trainer.setIndefiniteTrainingEnd();
		trainer.setProbingTime(10000);

		System.out.println("Starting Training of NN!");
		System.out.println("Maximum Trainings cycles: " + trainer.getMaximumTrainingCycles());
		System.out.println("Maximum error: " + trainer.getMaximumError());
		printExpected();
		System.out.println("Start.");

		trainer.train(neuralNetwork);

		System.out.println("[0, 0] => [" + Math.round(neuralNetwork.feedForward(new Double[] { 0.0, 0.0 })[0]) + "]");
		System.out.println("[0, 1] => [" + Math.round(neuralNetwork.feedForward(new Double[] { 0.0, 1.0 })[0]) + "]");
		System.out.println("[1, 0] => [" + Math.round(neuralNetwork.feedForward(new Double[] { 1.0, 0.0 })[0]) + "]");
		System.out.println("[1, 1] => [" + Math.round(neuralNetwork.feedForward(new Double[] { 1.0, 1.0 })[0]) + "]");
	}

//	/**
//	 * The main entry point for all JavaFX applications.
//	 * The start method is called after the init method has returned,
//	 * and after the system is ready for the application to begin running.
//	 * <p>
//	 * <p>
//	 * NOTE: This method is called on the JavaFX Application Thread.
//	 * </p>
//	 *
//	 * @param primaryStage the primary stage for this application, onto which
//	 *                     the application scene can be set. The primary stage will be embedded in
//	 *                     the browser if the application was launched as an applet.
//	 *                     Applications may create other stages, if needed, but they will not be
//	 *                     primary stages and will not be embedded in the browser.
//	 */
//	@Override
//	public void start(final Stage primaryStage) throws Exception {
//
//		Group root = new Group();
//
//		Scene scene = new Scene(root, windowWidth, windowHeight, true, SceneAntialiasing.BALANCED);
//		scene.setFill(Color.LIGHTBLUE);
//
//		pixels = new Rectangle[pixelDimensions][pixelDimensions];
//		int rectangleWidth = windowWidth / pixelDimensions;
//		int rectangleHeight = windowHeight / pixelDimensions;
//
//		VBox outer = new VBox();
//		for (int i = 0; i < pixels.length; i++) {
//			Rectangle[] currentRow = pixels[i];
//			HBox currentRowBox = new HBox();
//			for (int j = 0; j < currentRow.length; j++) {
//				Rectangle rectangle = new Rectangle(rectangleWidth, rectangleHeight);
//				rectangle.setFill(Color.gray(ThreadLocalRandom.current().nextDouble(0, 1)));
//				pixels[i][j] = rectangle;
//				currentRowBox.getChildren().add(rectangle);
//			}
//			outer.getChildren().add(currentRowBox);
//		}
//		root.getChildren().add(outer);
//
//		Stage stage = new Stage();
//		stage.setScene(scene);
//		stage.setWidth(windowWidth);
//		stage.setHeight(windowHeight);
//
//		stage.show();
//
//		applyPixels();
//
//		new Thread(this::run).start();
//	}

	public static void main(String[] args) {
		new Example().run();
	}

	private class CanvasProbingConsumer implements BiConsumer<NeuralNetwork, NeuralNetworkTrainer> {

		CanvasProbingConsumer() {
		}

		/**
		 * Performs this operation on the given arguments.
		 *
		 * @param neuralNetwork the first input argument
		 * @param trainer       the second input argument
		 */
		@Override
		public void accept(final NeuralNetwork neuralNetwork, final NeuralNetworkTrainer trainer) {
			NeuralNetworkTrainer.DEFAULT_PROBING_CONSUMER.accept(neuralNetwork, trainer);
//			applyPixels();
		}
	}
}
