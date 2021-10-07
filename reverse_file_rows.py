read_file_path = "micro_emini_sp_500.csv"
write_file_path = "reversed_micro_emini_sp_500_historical.csv"
with open(read_file_path) as data_file:
    rows_read = [row.rstrip() for row in data_file]
print(len(rows_read))
with open(write_file_path, "a") as reversed_file:
    for row in range(len(rows_read) - 1, 0, -1):
        reversed_file.write(rows_read[row] + '\n')