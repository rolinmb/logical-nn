#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 8
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.1
#define EPOCHS 1000

typedef struct {
  double weights[HIDDEN_SIZE][INPUT_SIZE];
  double biases[HIDDEN_SIZE];
} hidden_layer;

typedef struct {
  double weights[OUTPUT_SIZE][HIDDEN_SIZE];
  double bias;
} output_layer;

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double sigmoid_deriv(double x) {
  return x * (1.0 - x);
}

void forward_hidden(double inputs[INPUT_SIZE], hidden_layer* hidden, double hidden_outputs[HIDDEN_SIZE]) {
  for (int i = 0; i < HIDDEN_SIZE; i++) {
	double weighted_sum = hidden->biases[i];
	for (int j = 0; j < INPUT_SIZE; j++) {
	  weighted_sum += inputs[i] * hidden->weights[i][j];
	}
	hidden_outputs[i] = sigmoid(weighted_sum);
  }
}

double forward_output(double hidden_outputs[HIDDEN_SIZE], output_layer* output) {
  double weighted_sum = output->bias;
  for (int i = 0; i < HIDDEN_SIZE; i++) {
	weighted_sum += hidden_outputs[i] * output->weights[0][i];
  }
  return sigmoid(weighted_sum);
}

void backpropagate(double inputs[INPUT_SIZE], hidden_layer* hidden, output_layer* output, double target) {
  double hidden_outputs[HIDDEN_SIZE];
  forward_hidden(inputs, hidden, hidden_outputs);
  double predicted_output = forward_output(hidden_outputs, output);
  double error = target - predicted_output;
  double output_delta = error * sigmoid_deriv(predicted_output);
  for (int i = 0; i < HIDDEN_SIZE; i++) {
	output->weights[0][i] += LEARNING_RATE * output_delta * hidden_outputs[i];
  }
  output->bias += LEARNING_RATE * output_delta;
  for (int i = 0; i < HIDDEN_SIZE; i++) {
	double hidden_delta = output_delta * output->weights[0][i] * sigmoid_deriv(hidden_outputs[i]);
	for (int j = 0; j < INPUT_SIZE; j++) {
	   hidden->weights[i][j] += LEARNING_RATE * hidden_delta * inputs[j];
	}
	hidden->biases[i] += LEARNING_RATE * hidden_delta;
  }
}