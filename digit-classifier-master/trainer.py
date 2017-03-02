import collect
import network

def main():
    training_data, validation_data, test_data = collect.load_mnist()
    net = network.NeuralNetwork([784, 50, 50, 10], 1.0, 16, 100)
    net.fit(training_data, validation_data)

if __name__ == '__main__':
  main()