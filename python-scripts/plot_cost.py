#!./bin/python3

import argparse
import matplotlib.pyplot as plt

import argparse
import matplotlib.pyplot as plt

def read_file_as_floats(file_name):
    with open(file_name, 'r') as file:
        return [float(line.strip()) for line in file]

def plot_data(data):
    plt.plot(data, 'o-')  # 'o-' is for line with circle markers
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Gradient')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot f64 values from a file.')
    parser.add_argument('file', type=str, help='Name of the file to read')

    args = parser.parse_args()

    data = read_file_as_floats(args.file)
    plot_data(data)

if __name__ == '__main__':
    main()
