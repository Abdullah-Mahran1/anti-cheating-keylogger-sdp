from Crypto.Cipher import AES
import binascii

# Key for decryption (128-bit)
key_hex = "000102030405060708090A0B0C0D0E0F"
ciphertext_hex = "D039E2D98BAB10FA55A6DFAF80C37178E62AE682DE81471BF40C105DABA08A5DECE18E1993EA5606060622882A2D7968FAA12C9BB8BFE10882DF3EC987AD26284968323CDE33B2B558D26FE2F66C28CA"
key = binascii.unhexlify(key_hex)
ciphertext = binascii.unhexlify(ciphertext_hex)

cipher = AES.new(key, AES.MODE_ECB)

# Decrypt the ciphertext
plaintext = cipher.decrypt(ciphertext)

# Remove padding
pad_length = plaintext[-1]
decrypted_text = plaintext[:-pad_length]

# Convert bytes to text (assuming UTF-8 encoding)
text = decrypted_text.decode("utf-8")

# Print the decrypted text
print("Decrypted text:", text)


