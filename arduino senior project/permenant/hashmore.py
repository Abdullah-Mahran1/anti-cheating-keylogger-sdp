import time
def write_to_microcontroller_file(ID, result_string):
    microcontroller_file_path = 'D:/arduino senior project/microcontrollerid.txt.txt'
    frommicro_file_path = 'D:/arduino senior project/frommicro.txt'


    # Read existing content from microcontroller file
    with open(microcontroller_file_path, 'r') as microcontroller_file:
        lines = microcontroller_file.readlines()


    # Update the entry for the provided ID
    updated_lines = [f"{ID},{result_string}\n" if line.startswith(f"{ID},") else line for line in lines]


    # If the ID is not found, add a new entry
    if not any(line.startswith(f"{ID},") for line in lines):
        updated_lines.append(f"{ID},{result_string}\n")


    # Write the updated content back to the microcontroller file
    with open(microcontroller_file_path, 'w') as microcontroller_file:
        microcontroller_file.writelines(updated_lines)


    # Clear the content of frommicro file
    #with open(frommicro_file_path, 'w') as frommicro_file:
       # frommicro_file.write("")


# Example usage
# write_to_microcontroller_file("1001", "202001804")




def process_microcontroller_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()


    result_string = ""
    ID = ""


    # Processing the lines and concatenating to result_string
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
        # Print or use the result_string and ID variables as needed
        print(result_string)


        # Check if result_string starts with !@ and ends with _+
        if result_string.startswith('!@') and result_string.endswith('_+'):
            result_string = result_string[2:-2]  # Remove !@ and _+
            print(result_string)


            # Now open the id.txt file and check if result_string is in the file
            with open('D:/arduino senior project/id.txt', 'r') as id_file:
                id_file_content = id_file.read()
                if result_string in id_file_content:
                    print("Result string is in id.txt")


                    # Call the function to write to microcontrollerid.txt
                    write_to_microcontroller_file(ID, result_string)
                else:
                    print("Result string is not in id.txt")
                    




        print("ID:", ID)


while True:
    with open('D:/arduino senior project/frommicro.txt', 'r') as frommicro_file:
        content = frommicro_file.read().strip()


    if content:
        process_microcontroller_data('D:/arduino senior project/frommicro.txt')
    else:
        time.sleep(1)  # Adjust the sleep time as needed