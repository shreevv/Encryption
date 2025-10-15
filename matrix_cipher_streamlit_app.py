import streamlit as st
import numpy as np
from deep_translator import GoogleTranslator
import random

# --- 1. SETUP: Define the character set and mappings ---
ALPHABET = 'abcdefghijklmnopqrstuvwxyz '
MODULUS = len(ALPHABET)
CHAR_TO_INT = {char: i for i, char in enumerate(ALPHABET)}
INT_TO_CHAR = {i: char for i, char in enumerate(ALPHABET)}

# --- 2. HELPER FUNCTIONS for math operations ---
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d, x, y

def mod_inverse(n, m):
    d, x, y = extended_gcd(n, m)
    if d != 1:
        return None
    return x % m

# --- 3. CORE CRYPTOGRAPHIC FUNCTIONS ---
def generate_invertible_key(size=2):
    while True:
        matrix = np.random.randint(0, MODULUS, size=(size, size))
        det = int(round(np.linalg.det(matrix))) % MODULUS
        if det != 0 and gcd(det, MODULUS) == 1:
            return matrix

def get_key_inverse(key_matrix):
    det = int(round(np.linalg.det(key_matrix)))
    det_inv = mod_inverse(det, MODULUS)
    
    if det_inv is None:
        raise ValueError("Matrix is not invertible")

    if key_matrix.shape == (2, 2):
        a, b = key_matrix[0, 0], key_matrix[0, 1]
        c, d = key_matrix[1, 0], key_matrix[1, 1]
        adjugate = np.array([[d, -b], [-c, a]])
        inverse_matrix = (det_inv * adjugate) % MODULUS
        return inverse_matrix
    else:
        adjugate = np.linalg.inv(key_matrix) * det
        inverse_matrix = (det_inv * adjugate) % MODULUS
        return np.round(inverse_matrix).astype(int)

def process_text(text, key_matrix, mode):
    # This function is now updated to return intermediate steps for the explanation
    if mode == 'decrypt':
        try:
            processing_key = get_key_inverse(key_matrix)
        except ValueError:
            return "Error: Key not invertible.", [], []
    else:
        processing_key = key_matrix
        
    text_clean = text.lower()
    text_clean = ''.join(filter(lambda char: char in ALPHABET, text_clean))
    
    block_size = processing_key.shape[0]
    if len(text_clean) % block_size != 0:
        padding = block_size - (len(text_clean) % block_size)
        text_clean += ALPHABET[-1] * padding

    vectors = []
    for i in range(0, len(text_clean), block_size):
        block = text_clean[i:i+block_size]
        vector = [CHAR_TO_INT[char] for char in block]
        vectors.append(np.array(vector))

    result_vectors = [(processing_key @ v) % MODULUS for v in vectors]

    result_text = ""
    for vec in result_vectors:
        for num in vec:
            result_text += INT_TO_CHAR[int(num)]
            
    return result_text.strip(), vectors, result_vectors

# --- 4. EXPLANATION FUNCTION ---
def display_working_details(text, key, vectors, result_vectors):
    st.subheader(f'Encryption Steps for: "{text}"')

    # Step 1: Text to Vectors
    st.markdown("##### 1. Convert Plaintext to Numerical Vectors")
    st.write("The text is processed in blocks of 2 characters. Each character is mapped to a number (a=0, ..., space=26).")
    cols = st.columns(len(vectors))
    for i, col in enumerate(cols):
        with col:
            chars = text.lower()[i*2:(i*2)+2]
            st.metric(label=f"'{chars}'", value=f"[{vectors[i][0]}, {vectors[i][1]}]")

    # Step 2 & 3: Matrix Multiplication and Modulo
    st.markdown(f"##### 2. Matrix Multiplication & Modulo {MODULUS}")
    st.write("Each numerical vector (P) is multiplied by the key matrix (K), and the result is taken modulo 27.")
    
    for i in range(len(vectors)):
        with st.expander(f"Calculation for vector {i+1}: [{vectors[i][0]}, {vectors[i][1]}]"):
            p_vec = vectors[i]
            c_vec_raw = np.array(key) @ p_vec
            c_vec_mod = result_vectors[i]
            
            st.latex(r'''
            \begin{pmatrix} %
            C_1 \\ C_2 
            \end{pmatrix}
            =
            \begin{pmatrix} %
            k_{11} & k_{12} \\ k_{21} & k_{22}
            \end{pmatrix}
            \begin{pmatrix} %
            P_1 \\ P_2
            \end{pmatrix}
            \pmod{27}
            ''')
            
            st.latex(fr'''
            \begin{pmatrix} %
            {c_vec_raw[0]} \\ {c_vec_raw[1]}
            \end{pmatrix}
            =
            \begin{pmatrix} %
            {key[0][0]} & {key[0][1]} \\ {key[1][0]} & {key[1][1]}
            \end{pmatrix}
            \begin{pmatrix} %
            {p_vec[0]} \\ {p_vec[1]}
            \end{pmatrix}
            \implies
            \begin{pmatrix} %
            {c_vec_raw[0]} \pmod{27} \\ {c_vec_raw[1]} \pmod{27}
            \end{pmatrix}
            =
            \begin{pmatrix} %
            {c_vec_mod[0]} \\ {c_vec_mod[1]}
            \end{pmatrix}
            ''')

    # Step 4: Vectors to Ciphertext
    st.markdown("##### 4. Convert Back to Text")
    st.write("Finally, the new numerical vectors are mapped back to characters to form the ciphertext.")
    cols = st.columns(len(result_vectors))
    for i, col in enumerate(cols):
        with col:
            chars = INT_TO_CHAR[result_vectors[i][0]] + INT_TO_CHAR[result_vectors[i][1]]
            st.metric(label=f"[{result_vectors[i][0]}, {result_vectors[i][1]}]", value=f"'{chars}'")

# --- 5. TRANSLATION FUNCTION ---
def translate_text(text, target_language='hi'):
    if not text:
        return ""
    return GoogleTranslator(source='auto', target=target_language).translate(text)

# --- 6. STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("Matrix Encryption & Translation Tool")

if 'encryption_key' not in st.session_state:
    st.session_state.encryption_key = generate_invertible_key()

st.sidebar.subheader("Current Encryption Key")
st.sidebar.write(st.session_state.encryption_key)
if st.sidebar.button("Generate New Key"):
    st.session_state.encryption_key = generate_invertible_key()
    st.sidebar.success("New key generated!")

st.header("Live Demonstration")
if st.button("Generate Random Example"):
    names = ["Vikas", "Shree Dutt", "Om", "Lakshay", "Priyansh"]
    activities = ["loves playing football", "is learning to code", "is reading a book", "designs a website", "is solving a puzzle"]
    sentence = f"{random.choice(names)} {random.choice(activities)}"
    st.session_state.demo_sentence = sentence
    
    encrypted, vectors, result_vectors = process_text(sentence, st.session_state.encryption_key, 'encrypt')
    st.session_state.demo_encrypted = encrypted
    st.session_state.demo_decrypted, _, _ = process_text(encrypted, st.session_state.encryption_key, 'decrypt')
    
    st.session_state.demo_vectors = vectors
    st.session_state.demo_result_vectors = result_vectors

if 'demo_sentence' in st.session_state:
    st.markdown("---")
    st.write("**Original Sentence:**")
    st.info(st.session_state.demo_sentence)
    st.write("**Encrypted (Ciphertext):**")
    st.code(st.session_state.demo_encrypted)
    st.write("**Decrypted (Plaintext):**")
    st.success(st.session_state.demo_decrypted)

    if st.button("Show Detailed Working"):
        st.session_state.show_working = not st.session_state.get('show_working', False)

    if st.session_state.get('show_working', False):
        display_working_details(
            st.session_state.demo_sentence,
            st.session_state.encryption_key,
            st.session_state.demo_vectors,
            st.session_state.demo_result_vectors
        )
    st.markdown("---")

st.header("Try It Yourself")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Encrypt")
    plaintext = st.text_area("Enter text to encrypt:", height=150, key="encrypt_input")
    if st.button("Encrypt"):
        if plaintext:
            ciphertext, _, _ = process_text(plaintext, st.session_state.encryption_key, 'encrypt')
            st.session_state.user_ciphertext = ciphertext
        else:
            st.warning("Please enter some text to encrypt.")
    if 'user_ciphertext' in st.session_state:
        st.write("### Encrypted Text:")
        st.code(st.session_state.user_ciphertext)
        st.write("### Translate Encrypted Text:")
        target_lang_encrypt = st.selectbox("Select language:", GoogleTranslator().get_supported_languages(as_dict=True).keys(), key='encrypt_lang')
        if st.button("Translate", key='translate_encrypt'):
            translated_ciphertext = translate_text(st.session_state.user_ciphertext, target_lang_encrypt)
            st.write(f"**Translated to {target_lang_encrypt}:**")
            st.info(translated_ciphertext)

with col2:
    st.subheader("Decrypt")
    ciphertext_to_decrypt = st.text_area("Enter text to decrypt:", height=150, key="decrypt_input")
    if st.button("Decrypt"):
        if ciphertext_to_decrypt:
            decrypted_text, _, _ = process_text(ciphertext_to_decrypt, st.session_state.encryption_key, 'decrypt')
            st.session_state.user_decryptedtext = decrypted_text
        else:
            st.warning("Please enter some text to decrypt.")
    if 'user_decryptedtext' in st.session_state:
        st.write("### Decrypted Text:")
        st.success(st.session_state.user_decryptedtext)
        st.write("### Translate Decrypted Text:")
        target_lang_decrypt = st.selectbox("Select language:", GoogleTranslator().get_supported_languages(as_dict=True).keys(), key='decrypt_lang')
        if st.button("Translate", key='translate_decrypt'):
            translated_decrypted_text = translate_text(st.session_state.user_decryptedtext, target_lang_decrypt)
            st.write(f"**Translated to {target_lang_decrypt}:**")
            st.info(translated_decrypted_text)
