def process_microcontroller_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    result_string = ""
    ID = ""
    password = ""

    # Processing the lines and extracting ID and password
    for line in lines:
        line = line.strip()

        # Check if the line contains "ID:"
        if "ID:" in line:
            ID = line.split(":")[1].strip()
        else:
            elements = line.split(',')
            if len(elements) in {3, 5}:
                before_last_element = elements[-2]
                result_string += before_last_element

    # Check if the result_string is exactly 12 characters long
    if len(result_string) != 13:
        print("Error: Result string should be exactly 13 characters long.")
        print(result_string)
        write_to_microcontroller_file(ID, "none")
    else:
        # Print or use the result_string, ID, and password variables as needed
        print("Result String:", result_string)

        # Check if result_string starts with !@ and ends with _+
        if result_string.startswith('!@') and result_string.endswith('_+'):
            result_string = result_string[2:-2]  # Remove !@ and _+
            print("Processed Result String:", result_string)

            # Extract ID and password from the processed result_string
            ID = result_string[:-4]
            password = result_string[-4:]

            print("ID:", ID)
            print("Password:", password)

            # Now open the id.txt file and check if ID is in the file
            with open('D:/arduino senior project/id.txt', 'r') as id_file:
                id_file_content = id_file.read()
                if ID in id_file_content:
                    print("ID is in id.txt")

                    # Call the function to write to microcontrollerid.txt
                    write_to_microcontroller_file(ID, password)
                else:
                    print("ID is not in id.txt")

while True:
    with open('D:/arduino senior project/frommicro.txt', 'r') as frommicro_file:
        content = frommicro_file.read().strip()

    if content:
        process_microcontroller_data('D:/arduino senior project/frommicro.txt')
    else:
        time.sleep(1)  # Adjust the sleep time as needed
