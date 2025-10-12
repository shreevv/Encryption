# matrix_cipher_streamlit_app.py
"""
Matrix Cipher Visualizer — Streamlit App

Run: pip install streamlit numpy matplotlib plotly
Then: streamlit run matrix_cipher_streamlit_app.py

Features:
 - Browser UI (opens automatically when you run with Streamlit)
 - Generate random demo sentences using provided names
 - Choose or generate an invertible key matrix (mod 26)
 - Step-by-step encryption & decryption with visuals
 - Heatmap of key matrix, 2D vector mapping (for 2x2), and animated step display
 - Dark glass-like theme via custom CSS

Author: ChatGPT (GPT-5 Thinking mini)
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from math import gcd
from random import choice, randint, sample
import time

# ---------- Utility functions (matrix modular arithmetic) ----------

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modinv(a, m):
    a = a % m
    g, x, _ = egcd(a, m)
    if g != 1:
        return None
    return x % m


def matrix_mod_det(mat, m):
    det = int(round(np.linalg.det(mat)))
    return det % m, det


def matrix_mod_inv(mat, m=26):
    n = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square')
    det = int(round(np.linalg.det(mat)))
    det_mod = det % m
    inv_det = modinv(det_mod, m)
    if inv_det is None:
        return None
    cof = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(mat, i, axis=0), j, axis=1)
            cof[i, j] = ((-1) ** (i + j)) * int(round(np.linalg.det(minor)))
    adj = cof.T
    inv_mat = (inv_det * adj) % m
    return inv_mat.astype(int)


def random_invertible_matrix(n, m=26, tries=200):
    for _ in range(tries):
        mat = np.random.randint(0, m, size=(n, n))
        det_mod, _ = matrix_mod_det(mat, m)
        if gcd(int(det_mod), m) == 1:
            return mat
    raise ValueError('Failed to find invertible matrix')

# ---------- Text-numeric mapping ----------

def text_to_numbers(text):
    nums = []
    for ch in text.upper():
        if 'A' <= ch <= 'Z':
            nums.append(ord(ch) - ord('A'))
    return nums


def numbers_to_text(nums):
    return ''.join(chr((n % 26) + ord('A')) for n in nums)


def chunkify(lst, size, pad_value=23):
    chunks = []
    for i in range(0, len(lst), size):
        chunk = lst[i:i+size]
        if len(chunk) < size:
            chunk += [pad_value] * (size - len(chunk))
        chunks.append(chunk)
    return chunks


def encrypt_block(block, key_mat, m=26):
    vec = np.array(block, dtype=int).reshape((-1, 1))
    product = key_mat.dot(vec)
    cipher = product % m
    return cipher.flatten().tolist(), product.flatten().tolist()


def decrypt_block(block, inv_key_mat, m=26):
    vec = np.array(block, dtype=int).reshape((-1, 1))
    product = inv_key_mat.dot(vec)
    plain = product % m
    return plain.flatten().tolist(), product.flatten().tolist()

# ---------- Random sentence generator ----------

NAMES = ["Shree Dutt", "Vikas Gora", "Priyansh Roa", "Lakshay", "Om"]
VERBS = ["loves", "likes", "hates", "plays", "studies", "enjoys"]
OBJECTS = ["football", "coding", "music", "college", "chess", "reading"]

SENTENCE_TEMPLATES = [
    "{name} {verb} {obj}.",
    "Did you know {name} {verb} {obj}?",
    "Everyone says {name} {verb} {obj} every weekend.",
]


def random_sentence():
    name = choice(NAMES)
    verb = choice(VERBS)
    obj = choice(OBJECTS)
    template = choice(SENTENCE_TEMPLATES)
    return template.format(name=name, verb=verb, obj=obj)

# ---------- Visualization helpers ----------

def heatmap_plotly(mat, title="Key Matrix (mod 26)"):
    z = (mat % 26).astype(int)
    fig = go.Figure(data=go.Heatmap(z=z, text=z, texttemplate="%{text}", colorscale='Viridis'))
    fig.update_layout(title=title, xaxis_title='Col', yaxis_title='Row', width=450, height=420)
    return fig


def vector_map_plotly(plain_blocks, cipher_blocks, title='Plain -> Cipher'):
    # Only for 2D blocks
    xs = []
    ys = []
    xe = []
    ye = []
    annotations = []
    for p, c in zip(plain_blocks, cipher_blocks):
        px, py = p[0], p[1]
        cx, cy = c[0], c[1]
        xs.append(px); ys.append(py)
        xe.append(cx); ye.append(cy)
        annotations.append(dict(x=px, y=py, text=f'P({px},{py})', showarrow=False))
        annotations.append(dict(x=cx, y=cy, text=f'C({cx},{cy})', showarrow=False))

    fig = go.Figure()
    for i in range(len(xs)):
        fig.add_trace(go.Scatter(x=[xs[i], xe[i]], y=[ys[i], ye[i]], mode='lines+markers'))
    fig.update_layout(title=title, xaxis=dict(range=[-1, 26]), yaxis=dict(range=[-1, 26]), annotations=annotations, width=600, height=500)
    fig.update_xaxes(title='Component 0')
    fig.update_yaxes(title='Component 1')
    return fig

# ---------- Streamlit App UI ----------

st.set_page_config(page_title="Matrix Cipher Visualizer", layout='wide', initial_sidebar_state='expanded')

# Custom CSS for nicer look
st.markdown("""
<style>
body {background: linear-gradient(135deg, #0f172a 0%, #020617 100%); color: #dbeafe}
.css-1v3fvcr {background: rgba(255,255,255,0.03);} /* card backgrounds */
.stButton>button {background-color:#312e81; color:white}
h1 {color: #a78bfa}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# ✨ Matrix Cipher Visualizer")
st.markdown("A neat visual demo of encryption/decryption using invertible matrices and modular arithmetic (Hill-style). Run, play, and learn!")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", ["Demo sentence", "Custom plaintext"], index=0)
    block_size = st.selectbox("Block size (n x n key)", [2, 3, 4], index=0)
    gen_key = st.checkbox("Generate random invertible key", value=True)
    if not gen_key:
        key_input_raw = st.text_area("Enter key matrix rows (comma-separated rows; spaces between numbers)\nExample for 2x2:\n3 3, 2 5", value="3 3, 2 5")
    modulus = st.number_input("Modulus", min_value=2, max_value=512, value=26)
    st.markdown("---")
    st.markdown("**Demo Options**")
    show_steps = st.checkbox("Show step-by-step animation", value=True)
    st.markdown("---")
    st.markdown("Built for demo — names included: Shree Dutt, Vikas Gora, Priyansh Roa, Lakshay, Om")

# Main UI columns
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("Plaintext")
    if mode == 'Demo sentence':
        if st.button('Generate random demo sentence'):
            demo_text = random_sentence()
            st.session_state['demo_text'] = demo_text
        plaintext = st.text_area("Plaintext (letters only will be used)", value=st.session_state.get('demo_text', random_sentence()), height=120)
    else:
        plaintext = st.text_area("Plaintext (letters only will be used)", value="HELLO THERE", height=120)

    st.subheader("Key Matrix")
    if gen_key:
        if 'key_mat' not in st.session_state or st.session_state.get('last_block') != block_size or st.session_state.get('last_mod') != modulus:
            key = random_invertible_matrix(block_size, modulus)
            st.session_state['key_mat'] = key
            st.session_state['last_block'] = block_size
            st.session_state['last_mod'] = modulus
        else:
            key = st.session_state['key_mat']
        if st.button('Regenerate random invertible key'):
            key = random_invertible_matrix(block_size, modulus)
            st.session_state['key_mat'] = key
    else:
        # parse input
        try:
            rows = [row.strip() for row in key_input_raw.split(',') if row.strip()]
            mat = [list(map(int, r.split())) for r in rows]
            key = np.array(mat, dtype=int)
            if key.shape[0] != key.shape[1] or key.shape[0] != block_size:
                st.warning('Parsed matrix size does not match selected block size — using random key instead')
                key = random_invertible_matrix(block_size, modulus)
        except Exception as e:
            st.warning('Could not parse key matrix — using random key instead')
            key = random_invertible_matrix(block_size, modulus)

    st.code(str(key.tolist()))

    # Perform encryption
    if st.button('Encrypt →'):
        # Process
        nums = text_to_numbers(plaintext)
        chunks = chunkify(nums, block_size)
        det_mod, det_raw = matrix_mod_det(key, modulus)
        inv_key = matrix_mod_inv(key, modulus)
        if inv_key is None:
            st.error(f'Key matrix not invertible modulo {modulus}. Determinant mod {modulus} = {det_mod}')
        else:
            st.success(f'Key is invertible mod {modulus}. Determinant raw={det_raw}, mod {modulus}={det_mod}')
            cipher_blocks = []
            products = []
            step_area = st.empty()
            for i, blk in enumerate(chunks):
                c, prod = encrypt_block(blk, key, modulus)
                cipher_blocks.append(c)
                products.append(prod)
                if show_steps:
                    with step_area.container():
                        st.markdown(f"**Block {i+1}**: Plain numbers: {blk}")
                        st.markdown(f"Matrix product (before mod {modulus}): {prod}")
                        st.markdown(f"After mod {modulus}: {c}")
                        st.progress(int((i+1)/len(chunks)*100))
                        time.sleep(0.35)
            flat_cipher = [x for b in cipher_blocks for x in b]
            cipher_text = numbers_to_text(flat_cipher)
            st.session_state['cipher_text'] = cipher_text
            st.session_state['cipher_blocks'] = cipher_blocks
            st.session_state['products'] = products
            st.session_state['inv_key'] = inv_key
            st.success('Encryption complete — ciphertext:')
            st.code(cipher_text)

with col2:
    st.subheader('Visuals & Analysis')
    st.plotly_chart(heatmap_plotly(key), use_container_width=True)
    det_mod, det_raw = matrix_mod_det(key, modulus)
    st.write(f'Determinant raw = {det_raw}, mod {modulus} = {det_mod}')

    if block_size == 2:
        # show vector map if ciphertext exists or show sample mapping
        sample_text = plaintext
        nums = text_to_numbers(sample_text)
        chunks = chunkify(nums, 2)
        plain_example = chunks[:6]
        cipher_example = []
        for blk in plain_example:
            c, _ = encrypt_block(blk, key, modulus)
            cipher_example.append(c)
        st.plotly_chart(vector_map_plotly(plain_example, cipher_example), use_container_width=True)

    st.markdown('---')
    st.subheader('Decryption')
    if 'cipher_text' in st.session_state:
        st.write('Ciphertext:', st.session_state['cipher_text'])
        if st.button('Decrypt ←'):
            inv_key = st.session_state.get('inv_key')
            if inv_key is None:
                # compute
                inv_key = matrix_mod_inv(key, modulus)
            cipher_blocks = st.session_state.get('cipher_blocks')
            recovered = []
            recovered_products = []
            for i, blk in enumerate(cipher_blocks):
                p, prod = decrypt_block(blk, inv_key, modulus)
                recovered.append(p)
                recovered_products.append(prod)
            flat_plain = [x for b in recovered for x in b]
            recovered_text = numbers_to_text(flat_plain)
            st.success('Decryption complete — recovered (A-Z):')
            st.code(recovered_text)
    else:
        st.info('Encrypt some text first to enable decryption demo')

st.markdown('---')
st.markdown('### About this demo')
st.markdown('This app demonstrates a Hill-style matrix cipher using modular arithmetic. It is educational — do not use for real cryptographic secrecy.')

# Footer: Tips
st.caption('Tip: Try block sizes 2 and 3 and see how padding affects results.')
