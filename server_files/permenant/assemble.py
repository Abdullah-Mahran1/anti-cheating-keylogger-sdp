def update_data(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        keystrokes = input_file.readlines()

    formatted_data = ""
    for i in range(0, len(keystrokes), 2):
        key = keystrokes[i].split()[0]
        if key == "SPACE":
            formatted_data += " "
        else:
            formatted_data += key

    with open(output_file_path, 'a') as output_file:
        output_file.write(formatted_data)

# Paths for the first student
input_file_path_1 = r"D:\arduino senior project\201804181\Studentkeystrocks.txt"
output_file_path_1 = r"D:\arduino senior project\201804181\StudentData.txt"

# Paths for the second student
input_file_path_2 = r"D:\arduino senior project\201906710\Studentkeystrocks.txt"
output_file_path_2 = r"D:\arduino senior project\201906710\StudentData.txt"

# Paths for the third student
input_file_path_3 = r"D:\arduino senior project\201912079\Studentkeystrocks.txt"
output_file_path_3 = r"D:\arduino senior project\201912079\StudentData.txt"

# Paths for the third student
input_file_path_4 = r"D:\arduino senior project\202001804\Studentkeystrocks.txt"
output_file_path_4 = r"D:\arduino senior project\202001804\StudentData.txt"

# Paths for the third student
input_file_path_5 = r"D:\arduino senior project\202001805\Studentkeystrocks.txt"
output_file_path_5 = r"D:\arduino senior project\202001805\StudentData.txt"

# Paths for the third student
input_file_path_6 = r"D:\arduino senior project\202001806\Studentkeystrocks.txt"
output_file_path_6 = r"D:\arduino senior project\202001806\StudentData.txt"

# Update data for the first student
update_data(input_file_path_1, output_file_path_1)

# Update data for the second student
update_data(input_file_path_2, output_file_path_2)

# Update data for the third student
update_data(input_file_path_3, output_file_path_3)

# Update data for the third student
update_data(input_file_path_4, output_file_path_4)

# Update data for the third student
update_data(input_file_path_5, output_file_path_5)

# Update data for the third student
update_data(input_file_path_6, output_file_path_6)