from Crypto.Cipher import AES
from Crypto.Hash import CMAC

def compute_cmac(input_file_path, key_hex):
    # Read the input file content
    with open(input_file_path, 'rb') as file:
        input_data = file.read()
    
    # Create a new CMAC object using AES
    key = bytes.fromhex(key_hex)
    cobj = CMAC.new(key, ciphermod=AES)

    # Update the CMAC object with the file content
    cobj.update(input_data)

    # Print the CMAC
    print("CMAC:", cobj.hexdigest())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compute_cmac.py <input_file_path> <key_hex>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    key_hex = sys.argv[2]
    compute_cmac(input_file_path, key_hex)
