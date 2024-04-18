import csv
import random

def generate_matrix_csv(n, m, k, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the size of the matrix
        writer.writerow([n])
        writer.writerow([m])
        writer.writerow([k])
        # Generate and write Matrix A
        row = [round(random.uniform(-100, 100), 6) for _ in range(n * m)]
        writer.writerow(row)
        # Generate and write Matrix B
        row = [round(random.uniform(-100, 100), 6) for _ in range(m * k)]
        writer.writerow(row)


scale = [1, 3, 10, 32, 100, 316, 1000, 3162, 10000, 311623, 100000]
# Example usage:
for index in range(4, 7):
    # adjustable parameters
    n = m = k = scale[index]
    exponent = index / 2
    file_path = f'/mnt/c/SUSCode/CS205_C++-Program-Design/project-3/data/mat_data_10^{exponent:.1f}.csv'
    generate_matrix_csv(n, m, k, file_path)
    print(f"CSV file '{file_path}' generated successfully.")
