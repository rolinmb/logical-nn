#include "main.h"

int main() {
  struct timeval stop, start;
  gettimeofday(&start, NULL);
  hidden_layer hiddenLayer;
  output_layer outputLayer;
  srand(time(0));
  for (int i = 0; i < HIDDEN_SIZE; i++) {
	for (int j = 0; j < INPUT_SIZE; j++) {
	  hiddenLayer.weights[i][j] = (double)rand() / RAND_MAX;
	}
	hiddenLayer.biases[i] = (double)rand() / RAND_MAX;
  }
  for (int i = 0; i < OUTPUT_SIZE; i++) {
	outputLayer.bias = (double)rand() / RAND_MAX;
	for (int j = 0; j < HIDDEN_SIZE; j++) {
	  outputLayer.weights[i][j] = (double)rand() / RAND_MAX;
	}
  }
  double training_data[][INPUT_SIZE] = {{0,0}, {0,1}, {1,0}, {1,1}};
  // double target[] = {0, 1, 1, 1}; // OR
  // double target[] = {0, 1, 1, 0}; // XOR
  // double target[] = {0, 0, 0, 1}; // AND
  double target[] = {1, 1, 1, 0}; // NAND
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
	double total_error = 0.0;
	for (int i = 0; i < 4; i++) {
	  double hidden_outputs[HIDDEN_SIZE];
	  forward_hidden(training_data[i], &hiddenLayer, hidden_outputs);
	  backpropagate(training_data[i], &hiddenLayer, &outputLayer, target[i]);
	  double error = target[i] - forward_output(hidden_outputs, &outputLayer);
	  total_error += error * error;
	}
	printf("Epoch %d -> Error: %lf\n", epoch, total_error);
  }
  printf("Testing:\n");
  for (int i = 0; i < 4; i++) {
	double hidden_outputs[HIDDEN_SIZE];
	forward_hidden(training_data[i], &hiddenLayer, hidden_outputs);
	double prediction = forward_output(hidden_outputs, &outputLayer);
	printf("Input: %d %d, Target: %lf, Predicted: %lf\n", (int)training_data[i][0], (int)training_data[i][1], target[i], prediction);
  }
  gettimeofday(&stop, NULL);
  unsigned int micro_scnds = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
  printf("\n[Total program execution time: %lu microseconds]\n",micro_scnds);
  return 0;
}