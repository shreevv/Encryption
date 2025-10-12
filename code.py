import numpy as np
import math

# --- 1. SETUP: Define the character set and mappings ---
ALPHABET = 'abcdefghijklmnopqrstuvwxyz ' # Using a space as a character
MODULUS = len(ALPHABET)
CHAR_TO_INT = {char: i for i, char in enumerate(ALPHABET)}
INT_TO_CHAR = {i: char for i, char in enumerate(ALPHABET)}

print(f"Alphabet: '{ALPHABET}'")
print(f"Modulus (alphabet size): {MODULUS}\n")

# --- 2. HELPER FUNCTIONS for math operations ---
def gcd(a, b):
    """Computes the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    """Extended Euclidean Algorithm to find modular inverse components."""
    if a == 0:
        return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d, x, y

def mod_inverse(n, m):
    """Finds the modular inverse of n modulo m."""
    d, x, y = extended_gcd(n, m)
    if d != 1:
        raise ValueError(f"Modular inverse does not exist for {n} mod {m}")
    return x % m

# --- 3. CORE CRYPTOGRAPHIC FUNCTIONS ---
def generate_invertible_key(size=2):
    """Generates a random invertible square matrix of a given size."""
    while True:
        matrix = np.random.randint(0, MODULUS, size=(size, size))
        det = int(round(np.linalg.det(matrix))) % MODULUS
        if det != 0 and gcd(det, MODULUS) == 1:
            return matrix

def get_key_inverse(key_matrix):
    """Calculates the modular inverse of a given key matrix."""
    det = int(round(np.linalg.det(key_matrix)))
    det_inv = mod_inverse(det, MODULUS)
    
    # For a 2x2 matrix [[a, b], [c, d]], the adjugate is [[d, -b], [-c, a]]
    if key_matrix.shape == (2, 2):
        a, b = key_matrix[0, 0], key_matrix[0, 1]
        c, d = key_matrix[1, 0], key_matrix[1, 1]
        adjugate = np.array([[d, -b], [-c, a]])
        inverse_matrix = (det_inv * adjugate) % MODULUS
        return inverse_matrix
    else:
        # A more general but complex method for larger matrices
        adjugate = np.linalg.inv(key_matrix) * det
        inverse_matrix = (det_inv * adjugate) % MODULUS
        return np.round(inverse_matrix).astype(int)

def process_text(text, key_matrix, mode):
    """
    Encrypts or decrypts text using the provided key matrix.
    'mode' can be 'encrypt' or 'decrypt'.
    """
    if mode == 'decrypt':
        # Use the inverse key for decryption
        processing_key = get_key_inverse(key_matrix)
    else:
        processing_key = key_matrix
        
    text = text.lower()
    # Filter out characters not in our alphabet
    text = ''.join(filter(lambda char: char in ALPHABET, text))
    
    # Pad text to be a multiple of the key size
    block_size = processing_key.shape[0]
    if len(text) % block_size != 0:
        padding = block_size - (len(text) % block_size)
        text += ALPHABET[-1] * padding # Pad with spaces

    # Convert text to numerical vectors
    result_text = ""
    for i in range(0, len(text), block_size):
        block = text[i:i+block_size]
        vector = np.array([CHAR_TO_INT[char] for char in block])
        
        # Perform matrix multiplication and modulo
        result_vector = (processing_key @ vector) % MODULUS
        
        # Convert result back to characters
        for num in result_vector:
            result_text += INT_TO_CHAR[int(num)]
            
    return result_text

# --- 4. DEMONSTRATION ---
# The message to be encrypted
plaintext = "vikas gora loves playing football"
print(f"Original Plaintext: '{plaintext}'")

# Generate a valid 2x2 key for encryption
encryption_key = generate_invertible_key(size=2)
print(f"\nGenerated Encryption Key:\n{encryption_key}")

# Encrypt the message
ciphertext = process_text(plaintext, encryption_key, mode='encrypt')
print(f"\nEncrypted Ciphertext: '{ciphertext}'")

# Decrypt the message
decrypted_text = process_text(ciphertext, encryption_key, mode='decrypt')
print(f"\nDecrypted Plaintext: '{decrypted_text.strip()}'") # .strip() removes padding
