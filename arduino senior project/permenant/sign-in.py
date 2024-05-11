import time
import hashlib
import os
from Crypto.Cipher import AES
import binascii
previous_modified_time = 0

def hash_password(password):
    # Create a new SHA-256 hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the password bytes
    hash_object.update(password.encode('utf-8'))

    # Get the hexadecimal representation of the hashed password
    hashed_password = hash_object.hexdigest()

    return hashed_password

# Example usage:

#--------------------
def write_to_microcontroller_file(ID, result_string):
    microcontroller_file_path = 'D:/arduino senior project/microcontrollerid.txt'
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
    # with open(frommicro_file_path, 'w') as frommicro_file:
    #     frommicro_file.write("")


# Example usage
# write_to_microcontroller_file("1001", "202001804")

def decrypt_file(file_path, key_hex):
    # Read the ciphertext from the file
    with open(file_path, "rb") as file:
        ciphertext_hex = file.read()

    # Convert the key from hexadecimal string to bytes
    key = binascii.unhexlify(key_hex)

    # Convert the ciphertext from hexadecimal string to bytes
    ciphertext = binascii.unhexlify(ciphertext_hex)

    # Create an AES cipher object with the key and ECB mode
    cipher = AES.new(key, AES.MODE_ECB)

    # Decrypt the ciphertext
    plaintext = cipher.decrypt(ciphertext)

    # Remove padding
    pad_length = plaintext[-1]
    decrypted_text = plaintext[:-pad_length]

    # Convert bytes to text (assuming UTF-8 encoding)
    text = decrypted_text.decode("utf-8")
    print("Decrypted text:", text)

    return text

# Usage example
# file_path = "D:/arduino senior project/frommicro.txt"  # Replace this with the actual file path
# key_hex = "000102030405060708090A0B0C0D0E0F"  # Replace this with the decryption key in hexadecimal format
# decrypted_text = decrypt_file(file_path, key_hex)
# print("Decrypted text:", decrypted_text)


def process_microcontroller_data(file_path):
    decrypted_text = decrypt_file(file_path, "000102030405060708090A0B0C0D0E0F")  # Replace the key with your decryption key

    result_string = ""
    ID = ""

    # Processing the lines and concatenating to result_string
    for line in decrypted_text.splitlines():
        line = line.strip()

        # Check if the line contains "ID:"
        if "ID:" in line:
            ID = line.split(":")[1].strip()
        elif "KeyDown" in line:
            elements = line.split(' ')
            if len(elements) in {3}:
                if "LSHIFTKEY" not in elements and "RSHIFTKEY" not in elements:
                    before_last_element = elements[-3]
                    result_string += before_last_element

    print("Result String:", result_string)

    # Check if the result_string is exactly 12 characters long
    if len(result_string) != 19:
        print("Error: Result string should be exactly 19 characters long.")
        write_to_microcontroller_file(ID, "none")
    else:
        # Print or use the result_string and ID variables as needed
        print("Processed Result:", result_string)

        # Check if result_string starts with !@ and ends with _+
        if result_string.startswith('!@') and result_string.endswith('_+'):
            result_string = result_string[2:-2]  # Remove !@ and _+
            print("Updated Result:", result_string)

            Userid = result_string[0:9]
            UserPass = result_string[11:15]
            print("User ID:", Userid)
            print("User Password:", UserPass)

            functionh = hash_password(UserPass)
            print("Hashed Password:", functionh)

            # Now open the id.txt file and check if result_string is in the file
            with open('D:/arduino senior project/id.txt', 'r') as id_file:
                id_file_content = id_file.readlines()

                for line in id_file_content:
                    elements = line.strip().split(',')
                    if len(elements) == 2:
                        current_userid, hash_value = elements
                        if current_userid == Userid:
                            print("ID is correct:", current_userid)
                            if functionh == hash_value:
                                print("Hash is correct:", hash_value)
                                write_to_microcontroller_file(ID, Userid)
                            else:
                                write_to_microcontroller_file(ID, "none")
                            break
                else:
                    print("UserID not found in id.txt")
                    write_to_microcontroller_file(ID, "none")

    print("ID:", ID)


while True:
    current_modified_time = os.path.getatime('D:/arduino senior project/frommicro.txt')
    if current_modified_time != previous_modified_time:
        with open('D:/arduino senior project/frommicro.txt', 'r') as frommicro_file:
            content = frommicro_file.read().strip()

        if content:
            process_microcontroller_data('D:/arduino senior project/frommicro.txt')
            # Clear the content of frommicro file
            with open('D:/arduino senior project/frommicro.txt', 'w') as frommicro_file:
                frommicro_file.write("")
        else:
            print("File is empty")
        previous_modified_time = current_modified_time

    time.sleep(1)  # Adjust the sleep time as needed
