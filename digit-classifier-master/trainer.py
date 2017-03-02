import collect
import network

def main():
    training_data, validation_data, test_data = collect.load_mnist()
    net = network.NeuralNetwork([784, 500, 500, 500, 10], 1.0, 100, 10000, 0)
    net.fit(training_data, validation_data, test_data)

if __name__ == '__main__':
  main()