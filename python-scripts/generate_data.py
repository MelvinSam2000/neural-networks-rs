#!./bin/python3

import csv
import random

def generate_data(n):
    data = []
    for _ in range(n):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        if x1**2 + x2**2 < 0.5**2:
            y = 1
        else:
            y = 0
        data.append([x1, x2, y])
    return data

def write_to_csv(file_name, data):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x1', 'x2', 'y'])  # Writing header
        writer.writerows(data)

data = generate_data(1000)
write_to_csv('../data/circle.csv', data)
