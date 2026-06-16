package NeuralNetwork;

public class NeuralNetwork {
    /**
     * The Layers of this Neural Network
     */
    Layer[] layers;
    /**
     * η - Learning rate of the Neural Network
     */
    double eta;
    /**
     * The last
     */
    double[] lastOutput;

    /**
     * Constructor of the Neural Network
     * @param layerSizes The size of each layer
     * @param eta The learning rate of the network - the smaller, the preciser but slow
     */
    public NeuralNetwork (int[] layerSizes, double eta) {
        if (layerSizes.length < 2 || eta <= 0) throw new IllegalArgumentException("Illegal Parameters!");

        layers = new Layer[layerSizes.length];

        layers[0] = new Layer(layerSizes[0]);
        for (int i = 1; i < layerSizes.length; i++) {
            layers[i] = new Layer(layerSizes[i], layers[i - 1].size);
        }

        this.eta = eta;
    }

    /**
     * Takes an input and calculates Output
     * @param inputs Activation of the input
     * @return Activations of the output layer
     */
    public double[] feedForward (double[] inputs) {
        if (inputs.length != layers[0].size) throw new IllegalArgumentException("Input Size does not match input layers size!");
        layers[0].activations = inputs;
        // For each Layer
        for (int i = 1; i < layers.length; i++) {
            // For each neuron
            double[] newActivations = new double[layers[i].size];
            for (int j = 0; j < layers[i].size; j++) {
                // Calculate activation
                newActivations[j] = sigmoid(dotProduct(layers[i - 1].activations, layers[i].weights[j]) + layers[i].biases[j]);
            }
            layers[i].activations = newActivations;
        }
        return lastOutput = layers[layers.length - 1].activations;
    }

    public void backPropagation(double[] expected) {
        int l = layers.length;
        for (int i = 0; i < layers[l - 1].size; i++) {
            layers[l - 1].deltas[i] = (layers[l - 1].activations[i] - expected[i]) * sigmoidDerivative(layers[l - 1].activations[i]);
        }
        for (int i = l - 2; i > 0; i--) {
            for (int j = 0; j < layers[i].size; j++) {
                double errorSum = 0;
                for (int k = 0; k < layers[i + 1].size; k++) {
                    errorSum += layers[i + 1].deltas[k] * layers[i + 1].weights[k][j];
                }
                layers[i].deltas[j] = errorSum * sigmoidDerivative(layers[i].activations[j]);
            }
        }

        updateWeightsAndBiases();
    }

    private void updateWeightsAndBiases() {
        for (int i = 1; i < layers.length; i++) {
            for (int j = 0; j < layers[i].size; j++) {
                layers[i].biases[j] -= eta * layers[i].deltas[j];

                for (int k = 0; k < layers[i - 1].size; k++) {
                    layers[i].weights[j][k] -= eta * layers[i].deltas[j] * layers[i - 1].activations[k];
                }
            }
        }
    }

    /**
     * Trains the neural network - Adjusts weights and biases
     * @param trainingData The training data as 2D array
     * @param expectedOutput The expected output as 2D array
     * @param epochs The number of epochs / repetitions
     */
    public void train (double[][] trainingData, double[][] expectedOutput, int epochs) {
        if (trainingData.length != expectedOutput.length) throw new IllegalArgumentException("Training size does not match expected size!");
        double averageCost = 0;
        for (int i = 0; i < epochs; i++) {
            double cost = 0;
            for (int j = 0; j < trainingData.length; j++) {
                feedForward(trainingData[j]);
                backPropagation(expectedOutput[j]);
                cost += cost(expectedOutput[j], layers[layers.length - 1].activations);
            }
            cost /= trainingData.length;
            averageCost += cost;
            System.out.println("Epoch " + i + " completed. Cost: " + cost);
        }
        averageCost /= epochs;
        System.out.println("Training finished. Average Cost: " + averageCost);
    }

    public double cost (double[] expected, double[] output) {
        if (expected.length != output.length) throw new IllegalArgumentException("Expected size does not match output size!");
        double d = 0;
        for (int i = 0; i < expected.length; i++) {
            d += Math.abs(expected[i] - output[i]) * Math.abs(expected[i] - output[i]);
        }
        return d;
    }
    public double cost (double[] expected) {
        return cost(expected, lastOutput);
    }

    private double sigmoid (double x) {
        double y;
        if( x < -10 )
            y = 0;
        else if( x > 10 )
            y = 1;
        else
            y = 1 / (1 + Math.exp(-x));
        return y;
    }
    private double sigmoidDerivative (double a) {
        return a * (1 - a);
    }

    private double dotProduct (double[] a, double[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Both Array a and b have to be the same size!");
        double product = 0;
        for (int i = 0; i < a.length; i++) {
            product += a[i] * b[i];
        }
        return product;
    }

    public double[][] getWeightsForLayer(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= layers.length) return null;
        return layers[layerIndex].weights;
    }
    public double[] getActivationsForLayer(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= layers.length) return null;
        return layers[layerIndex].activations;
    }
}
