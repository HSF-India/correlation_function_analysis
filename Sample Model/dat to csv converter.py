import csv

input_file = 'pscalar_0p0018.dat'
output_file = 'pscalar_0p0018.csv'

with open(input_file, 'r') as dt_file:
    dt_data = dt_file.readlines()

csv_data = [line.strip().split() for line in dt_data]

with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)

print(f"File '{output_file}' has been created successfully.")
