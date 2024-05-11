# from Crypto.Cipher import AES
# import binascii
# import time

# key_hex = "000102030405060708090A0B0C0D0E0F"

# def decrypt_file(file_path, key_hex):
#     # Read the ciphertext from the file
#     with open(file_path, "rb") as file:
#         ciphertext_hex = file.read()

#     # Convert the key from hexadecimal string to bytes
#     key = binascii.unhexlify(key_hex)

#     # Convert the ciphertext from hexadecimal string to bytes
#     ciphertext = binascii.unhexlify(ciphertext_hex)

#     # Create an AES cipher object with the key and ECB mode
#     cipher = AES.new(key, AES.MODE_ECB)

#     # Decrypt the ciphertext
#     plaintext = cipher.decrypt(ciphertext)

#     # Remove padding
#     pad_length = plaintext[-1]
#     decrypted_text = plaintext[:-pad_length]

#     # Convert bytes to text (assuming UTF-8 encoding)
#     text = decrypted_text.decode("utf-8")

#     return text.strip()  # Remove trailing whitespace

# def monitor_decrypt_save(file_path, key_hex, output_file):
#     while True:
#         try:
#             decrypted_text = decrypt_file(file_path, key_hex)
#             if decrypted_text:  # Only write if there's decrypted text
#                 with open(output_file, "a") as output:
#                     output.write(decrypted_text + "\n")
            
#             # Clear the encrypted data file
#             with open(file_path, "wb") as file:
#                 file.write(b"")
#         except Exception as e:
#             print("Error:", e)
        
#         time.sleep(10)  # Monitor every 10 seconds

# file_path = "D:/arduino senior project/202001804/StudentEncriptedData.txt"
# output_file = "D:/arduino senior project/202001804/Studentkeystrocks.txt"
# monitor_decrypt_save(file_path, key_hex, output_file)


from Crypto.Cipher import AES
import binascii
import time

key_hex = "000102030405060708090A0B0C0D0E0F"

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

    return text.strip()  # Remove trailing whitespace

def monitor_decrypt_save(file_paths, key_hex, output_files):
    while True:
        for i, file_path in enumerate(file_paths):
            try:
                decrypted_text = decrypt_file(file_path, key_hex)
                if decrypted_text:  # Only write if there's decrypted text
                    with open(output_files[i], "a") as output:
                        output.write(decrypted_text + "\n")

                # Clear the encrypted data file
                with open(file_path, "wb") as file:
                    file.write(b"")
            except Exception as e:
                print("Error:", e)

        time.sleep(10)  # Monitor every 10 seconds

# List of file paths and output file paths
file_paths = [
    "D:/arduino senior project/202001805/StudentEncriptedData.txt",
    "D:/arduino senior project/202001804/StudentEncriptedData.txt",
    "D:/arduino senior project/202001806/StudentEncriptedData.txt",
    "D:/arduino senior project/201804181/StudentEncriptedData.txt",
    "D:/arduino senior project/201906710/StudentEncriptedData.txt",
    "D:/arduino senior project/201912079/StudentEncriptedData.txt"
]
output_files = [
    "D:/arduino senior project/202001805/Studentkeystrocks.txt",
    "D:/arduino senior project/202001804/Studentkeystrocks.txt",
    "D:/arduino senior project/202001806/Studentkeystrocks.txt",
    "D:/arduino senior project/201804181/Studentkeystrocks.txt",
    "D:/arduino senior project/201906710/Studentkeystrocks.txt",
    "D:/arduino senior project/201912079/Studentkeystrocks.txt"
]

monitor_decrypt_save(file_paths, key_hex, output_files)