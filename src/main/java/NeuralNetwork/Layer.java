package NeuralNetwork;

import java.util.Random;

public class Layer {
    /**
     * The number of neurons this Layer contains
     */
    final int size;

    /**
     * The biases for each neuron in this layer
     */
    double[] biases;
    /**
     * The weights connecting each neuron of this layer with those of the previous layer
     */
    double[][] weights;
    /**
     * The activation value of each neuron in this layer
     */
    double[] activations;
    double[] deltas;

    Random rand;


    /**
     * Constructor for a layer of the Neural Network
     * @param neurons The number of neurons this layer should have
     * @param prevNeurons The number of neurons the previous layer had
     */
    public Layer (int neurons, int prevNeurons) {
        if (neurons <= 0 || prevNeurons <= 0) throw new IllegalArgumentException("Neurons must be greater than 0!");
        this.size = neurons;
        biases = new double[neurons];
        weights = new double[neurons][prevNeurons];
        activations = new double[neurons];
        deltas = new double[neurons];
        rand = new Random();
        randomize();
    }

    /**
     * Constructor for the Input Layer without the parameter prevNeurons
     * @param neurons The number of neurons of this input layer
     */
    public Layer (int neurons) {
        if (neurons <= 0) throw new IllegalArgumentException("Neurons must be greater than 0!");
        this.size = neurons;
        activations = new double[neurons];
        deltas = new double[neurons];
    }

    /**
     * Randomizes the biases and weights
     */
    private void randomize() {
        for (int i = 0; i < biases.length; i++) {
            biases[i] = randDouble();
        }
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = randDouble();
            }
        }
    }

    /**
     * @return A double between in the range [-1,1]
     */
    private double randDouble() {
        return rand.nextDouble() * 2 - 1;
    }
}