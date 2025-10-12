import flask
import webbrowser
import threading
import random
import numpy as np
import math
import json

# ==============================================================================
# 1. CRYPTOGRAPHIC CORE LOGIC (HILL CIPHER)
# ==============================================================================

# Define the character set. A prime modulus (27) simplifies finding invertible keys.
ALPHABET = 'abcdefghijklmnopqrstuvwxyz '
MODULUS = len(ALPHABET)
CHAR_TO_INT = {char: i for i, char in enumerate(ALPHABET)}
INT_TO_CHAR = {i: char for i, char in enumerate(ALPHABET)}

def gcd(a, b):
    """Computes the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    """Extended Euclidean Algorithm to find modular inverse."""
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
        # The inverse does not exist
        return None
    return x % m

def generate_invertible_key(size=2):
    """Generates a random invertible square matrix of a given size."""
    while True:
        # Create a random matrix
        matrix = np.random.randint(0, MODULUS, size=(size, size))
        # Check if its determinant is coprime to the modulus
        det = int(round(np.linalg.det(matrix))) % MODULUS
        if det != 0 and gcd(det, MODULUS) == 1:
            return matrix.tolist()

def get_key_inverse(key):
    """Calculates the modular inverse of a given key matrix."""
    key_matrix = np.array(key)
    det = int(round(np.linalg.det(key_matrix)))
    
    det_inv = mod_inverse(det, MODULUS)
    if det_inv is None:
        raise ValueError("Matrix is not invertible")

    # For a 2x2 matrix [[a, b], [c, d]], the inverse is det_inv * [[d, -b], [-c, a]]
    if key_matrix.shape == (2, 2):
        a, b = key_matrix[0, 0], key_matrix[0, 1]
        c, d = key_matrix[1, 0], key_matrix[1, 1]
        
        adjugate = np.array([[d, -b], [-c, a]])
        inverse_matrix = (det_inv * adjugate) % MODULUS
        return inverse_matrix.tolist()
    
    # General case for larger matrices (though we primarily use 2x2)
    adjugate = np.linalg.inv(key_matrix) * det
    inverse_matrix = (det_inv * adjugate) % MODULUS
    return np.round(inverse_matrix).astype(int).tolist()


def process_text(text, key, mode):
    """
    Core function to encrypt or decrypt text.
    'mode' can be 'encrypt' or 'decrypt'.
    """
    key_matrix = np.array(key)
    if mode == 'decrypt':
        key_matrix = np.array(get_key_inverse(key))
        
    text = text.lower()
    # Filter out characters not in our alphabet
    text = ''.join(filter(lambda char: char in ALPHABET, text))
    
    # Pad text to be a multiple of the key size
    block_size = key_matrix.shape[0]
    if len(text) % block_size != 0:
        padding = block_size - (len(text) % block_size)
        text += ALPHABET[-1] * padding # Pad with space

    # Convert text to numerical vectors
    vectors = []
    for i in range(0, len(text), block_size):
        block = text[i:i+block_size]
        vector = [CHAR_TO_INT[char] for char in block]
        vectors.append(np.array(vector))

    # Perform matrix multiplication
    result_vectors = [(key_matrix @ v) % MODULUS for v in vectors]

    # Convert result back to text
    result_text = ""
    for vec in result_vectors:
        for num in vec:
            result_text += INT_TO_CHAR[int(num)]
            
    return result_text, vectors, result_vectors

# ==============================================================================
# 2. FLASK WEB SERVER
# ==============================================================================

app = flask.Flask(__name__)

# --- HTML, CSS, and JavaScript Template ---
# This is served as a single page application.
# The styling is inspired by the Groww financial app (dark theme, clean layout).
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Matrix Encryption Demonstrator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #171c26;
            color: #e0e0e0;
        }
        .groww-card {
            background-color: #252a34;
            border: 1px solid #333944;
            border-radius: 12px;
        }
        .groww-input {
            background-color: #171c26;
            border: 1px solid #333944;
            color: #e0e0e0;
            border-radius: 8px;
            padding: 10px 14px;
            width: 100%;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .groww-input:focus {
            outline: none;
            border-color: #00b386;
            box-shadow: 0 0 0 2px rgba(0, 179, 134, 0.3);
        }
        .groww-btn {
            background-color: #00b386;
            color: #ffffff;
            font-weight: 600;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .groww-btn:hover {
            background-color: #009a72;
        }
        .groww-btn-secondary {
            background-color: transparent;
            color: #00b386;
            border: 1px solid #00b386;
        }
        .groww-btn-secondary:hover {
            background-color: rgba(0, 179, 134, 0.1);
        }
        .matrix-display {
            display: grid;
            grid-template-columns: repeat(2, 35px);
            gap: 8px;
            justify-content: center;
            padding: 16px;
            background-color: #171c26;
            border-radius: 8px;
        }
        .matrix-cell {
            background-color: #333944;
            height: 35px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            font-weight: 500;
        }
        .working-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s, visibility 0.3s;
        }
        .working-modal.visible {
            opacity: 1;
            visibility: visible;
        }
    </style>
</head>
<body class="p-4 md:p-8">

    <!-- Main Container -->
    <div class="max-w-4xl mx-auto">
        
        <!-- Header -->
        <header class="flex items-center space-x-4 mb-8">
            <div class="p-3 bg-green-500/10 rounded-lg">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#00b386" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-shield"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            </div>
            <div>
                <h1 class="text-2xl md:text-3xl font-bold text-white">Matrix Encryption</h1>
                <p class="text-sm md:text-base text-gray-400">A Visual Guide to the Hill Cipher Algorithm</p>
            </div>
        </header>

        <!-- Main Content -->
        <main class="space-y-6">

            <!-- Demo Section -->
            <section class="groww-card p-6">
                <h2 class="text-xl font-semibold text-white mb-4">Live Demonstration</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Controls -->
                    <div class="space-y-4">
                        <button id="generate-btn" class="groww-btn w-full">Generate Random Example</button>
                        <div class="text-center">
                            <p class="text-sm text-gray-400 mb-2">Using 2x2 Key Matrix</p>
                            <div id="key-matrix" class="matrix-display mx-auto">
                                <div class="matrix-cell">?</div><div class="matrix-cell">?</div>
                                <div class="matrix-cell">?</div><div class="matrix-cell">?</div>
                            </div>
                        </div>
                        <button id="show-working-btn" class="groww-btn groww-btn-secondary w-full" disabled>Show Detailed Working</button>
                    </div>

                    <!-- Results -->
                    <div class="space-y-4">
                        <div>
                            <label class="text-sm font-medium text-gray-400">Original Sentence</label>
                            <p id="original-text" class="p-3 bg-gray-800 rounded-md mt-1 min-h-[44px] break-words">Click 'Generate' to start.</p>
                        </div>
                        <div>
                            <label class="text-sm font-medium text-gray-400">Encrypted (Ciphertext)</label>
                            <p id="encrypted-text" class="p-3 bg-gray-800 rounded-md mt-1 min-h-[44px] font-mono break-words"></p>
                        </div>
                        <div>
                            <label class="text-sm font-medium text-gray-400">Decrypted (Plaintext)</label>
                            <p id="decrypted-text" class="p-3 bg-gray-800 rounded-md mt-1 min-h-[44px] break-words"></p>
                        </div>
                    </div>
                </div>
            </section>

            <!-- User Input Section -->
            <section class="groww-card p-6">
                <h2 class="text-xl font-semibold text-white mb-4">Try It Yourself</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <!-- Encrypt -->
                    <div class="space-y-4">
                        <h3 class="font-medium text-white">Encrypt Text</h3>
                        <textarea id="user-encrypt-input" class="groww-input" rows="3" placeholder="Enter plaintext here..."></textarea>
                        <button id="user-encrypt-btn" class="groww-btn">Encrypt</button>
                        <div>
                            <label class="text-sm font-medium text-gray-400">Result</label>
                            <p id="user-encrypt-output" class="p-3 bg-gray-800 rounded-md mt-1 min-h-[44px] font-mono break-words"></p>
                        </div>
                    </div>
                    <!-- Decrypt -->
                    <div class="space-y-4">
                        <h3 class="font-medium text-white">Decrypt Text</h3>
                        <textarea id="user-decrypt-input" class="groww-input" rows="3" placeholder="Enter ciphertext here..."></textarea>
                        <button id="user-decrypt-btn" class="groww-btn">Decrypt</button>
                        <div>
                            <label class="text-sm font-medium text-gray-400">Result</label>
                            <p id="user-decrypt-output" class="p-3 bg-gray-800 rounded-md mt-1 min-h-[44px] break-words"></p>
                        </div>
                    </div>
                </div>
                 <p class="text-xs text-center text-gray-500 mt-6">Note: Decryption requires the exact same key matrix used for encryption. This demo uses the key shown in the 'Live Demonstration' section.</p>
            </section>
        </main>

    </div>

    <!-- Working Modal -->
    <div id="working-modal" class="working-modal">
        <div class="groww-card w-11/12 max-w-3xl max-h-[90vh] overflow-y-auto relative p-6 md:p-8">
            <button id="close-modal-btn" class="absolute top-4 right-4 text-gray-400 hover:text-white">&times;</button>
            <div id="working-content">
                <!-- Detailed steps will be loaded here -->
            </div>
        </div>
    </div>
    
    <script>
        // Global state to hold the current key
        let currentKey = null;
        let currentDemoText = "";

        // DOM Elements
        const generateBtn = document.getElementById('generate-btn');
        const showWorkingBtn = document.getElementById('show-working-btn');
        const keyMatrixDiv = document.getElementById('key-matrix');
        const originalTextP = document.getElementById('original-text');
        const encryptedTextP = document.getElementById('encrypted-text');
        const decryptedTextP = document.getElementById('decrypted-text');
        
        const userEncryptInput = document.getElementById('user-encrypt-input');
        const userEncryptBtn = document.getElementById('user-encrypt-btn');
        const userEncryptOutput = document.getElementById('user-encrypt-output');
        
        const userDecryptInput = document.getElementById('user-decrypt-input');
        const userDecryptBtn = document.getElementById('user-decrypt-btn');
        const userDecryptOutput = document.getElementById('user-decrypt-output');
        
        const workingModal = document.getElementById('working-modal');
        const closeModalBtn = document.getElementById('close-modal-btn');
        const workingContent = document.getElementById('working-content');

        function updateKeyMatrixUI(key) {
            keyMatrixDiv.innerHTML = `
                <div class="matrix-cell">${key[0][0]}</div><div class="matrix-cell">${key[0][1]}</div>
                <div class="matrix-cell">${key[1][0]}</div><div class="matrix-cell">${key[1][1]}</div>
            `;
        }

        // --- Event Listeners ---
        
        generateBtn.addEventListener('click', async () => {
            originalTextP.textContent = "Generating...";
            encryptedTextP.textContent = "";
            decryptedTextP.textContent = "";
            generateBtn.disabled = true;
            
            try {
                const response = await fetch('/api/generate-demo');
                const data = await response.json();
                
                currentKey = data.key_matrix;
                currentDemoText = data.original;
                
                updateKeyMatrixUI(data.key_matrix);
                originalTextP.textContent = data.original;
                encryptedTextP.textContent = data.encrypted;
                decryptedTextP.textContent = data.decrypted;
                
                showWorkingBtn.disabled = false;
            } catch (error) {
                originalTextP.textContent = "Error generating demo. Please try again.";
                console.error("Error:", error);
            } finally {
                generateBtn.disabled = false;
            }
        });

        userEncryptBtn.addEventListener('click', async () => {
            const text = userEncryptInput.value;
            if (!text || !currentKey) {
                alert("Please generate a demo first to get a key, and enter some text to encrypt.");
                return;
            }
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text, key: currentKey, mode: 'encrypt' })
            });
            const data = await response.json();
            userEncryptOutput.textContent = data.result;
        });

        userDecryptBtn.addEventListener('click', async () => {
            const text = userDecryptInput.value;
            if (!text || !currentKey) {
                alert("Please generate a demo first to get a key, and enter some text to decrypt.");
                return;
            }
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text, key: currentKey, mode: 'decrypt' })
            });
            const data = await response.json();
            userDecryptOutput.textContent = data.result;
        });
        
        showWorkingBtn.addEventListener('click', async () => {
            workingContent.innerHTML = '<p class="text-center">Loading detailed steps...</p>';
            workingModal.classList.add('visible');
            
            const response = await fetch('/api/get-working', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ text: currentDemoText, key: currentKey })
            });
            const data = await response.json();
            workingContent.innerHTML = data.html;
        });
        
        closeModalBtn.addEventListener('click', () => {
            workingModal.classList.remove('visible');
        });
        
        // Close modal on escape key
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                workingModal.classList.remove('visible');
            }
        });
    </script>
</body>
</html>
"""

# --- Flask API Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    return HTML_TEMPLATE

@app.route('/api/generate-demo', methods=['GET'])
def generate_demo():
    """Generates a random sentence, key, and performs encryption/decryption."""
    names = ["Shree Dutt", "Vikas Gora", "Priyansh Roa", "Lakshay", "Om"]
    activities = ["loves playing football", "is learning to code", "is reading a book", "designs a website", "is solving a puzzle"]
    sentence = f"{random.choice(names)} {random.choice(activities)}"
    
    key = generate_invertible_key(size=2)
    encrypted, _, _ = process_text(sentence, key, 'encrypt')
    decrypted, _, _ = process_text(encrypted, key, 'decrypt')
    
    return flask.jsonify({
        "original": sentence,
        "key_matrix": key,
        "encrypted": encrypted,
        "decrypted": decrypted.strip() # Remove padding
    })

@app.route('/api/process', methods=['POST'])
def process_user_text():
    """Handles user-submitted text for encryption or decryption."""
    data = flask.request.json
    text = data.get('text')
    key = data.get('key')
    mode = data.get('mode')
    
    result, _, _ = process_text(text, key, mode)
    if mode == 'decrypt':
        result = result.strip()
    
    return flask.jsonify({"result": result})

@app.route('/api/get-working', methods=['POST'])
def get_working_details():
    """Generates a detailed HTML breakdown of the encryption process."""
    data = flask.request.json
    text = data.get('text')
    key = data.get('key')
    
    # Run encryption again to get intermediate steps
    encrypted, vectors, result_vectors = process_text(text, key, 'encrypt')
    
    # Build HTML string for explanation
    html = f'<h2 class="text-2xl font-bold text-white mb-4">Encryption Steps for: "{text}"</h2>'
    
    # Step 1: Text to Vectors
    html += '<h3 class="text-lg font-semibold text-green-400 mt-6 mb-2">1. Convert Plaintext to Numerical Vectors</h3>'
    html += '<p class="text-gray-400 mb-4">The plaintext is processed in blocks of 2 characters. Each character is mapped to a number (a=0, b=1, ..., space=26). If the text length is odd, it\'s padded with a space.</p>'
    
    vector_html = '<div class="flex flex-wrap gap-4">'
    for i, v in enumerate(vectors):
        chars = text[i*2:(i*2)+2]
        vector_html += f"""
        <div class="text-center p-3 bg-gray-800 rounded-lg">
            <p class="font-mono">'{chars}'</p>
            <p class="text-sm text-gray-400">&#x2193;</p>
            <p class="font-bold text-xl">[{v[0]}, {v[1]}]</p>
        </div>
        """
    vector_html += '</div>'
    html += vector_html

    # Step 2: Matrix Multiplication
    html += '<h3 class="text-lg font-semibold text-green-400 mt-6 mb-2">2. Matrix Multiplication</h3>'
    html += '<p class="text-gray-400 mb-4">Each numerical vector (P) is multiplied by the key matrix (K). Ciphertext Vector C = K &times; P.</p>'
    
    multiplication_html = '<div class="flex flex-wrap gap-4">'
    for i in range(len(vectors)):
        p_vec = vectors[i]
        c_vec = np.array(key) @ p_vec
        multiplication_html += f"""
        <div class="p-4 bg-gray-800 rounded-lg">
            <p class="font-mono text-center">
                <span class="p-2 bg-gray-900 rounded">K</span> &times; <span class="p-2 bg-gray-900 rounded">P<sub>{i+1}</sub></span>
            </p>
            <div class="flex items-center justify-center gap-4 mt-2">
                <div class="text-center">
                    <p class="font-mono">[[{key[0][0]}, {key[0][1]}],</p>
                    <p class="font-mono">[{key[1][0]}, {key[1][1]}]]</p>
                </div>
                <p>&times;</p>
                <div class="text-center">
                    <p class="font-mono">[{p_vec[0]}]</p>
                    <p class="font-mono">[{p_vec[1]}]</p>
                </div>
                <p>=</p>
                <div class="text-center">
                    <p class="font-mono">[{c_vec[0]}]</p>
                    <p class="font-mono">[{c_vec[1]}]</p>
                </div>
            </div>
        </div>
        """
    multiplication_html += '</div>'
    html += multiplication_html
    
    # Step 3: Modulo Operation
    html += f'<h3 class="text-lg font-semibold text-green-400 mt-6 mb-2">3. Apply Modulo {MODULUS}</h3>'
    html += f'<p class="text-gray-400 mb-4">Each element of the resulting vectors is taken modulo {MODULUS} to bring it back into the range of our alphabet.</p>'
    
    modulo_html = '<div class="flex flex-wrap gap-4">'
    for i in range(len(result_vectors)):
        c_vec_raw = np.array(key) @ vectors[i]
        c_vec_mod = result_vectors[i]
        modulo_html += f"""
        <div class="p-4 bg-gray-800 rounded-lg text-center">
            <p class="font-mono">[{c_vec_raw[0]}, {c_vec_raw[1]}] mod {MODULUS}</p>
            <p class="text-sm text-gray-400">&#x2193;</p>
            <p class="font-bold text-xl">[{c_vec_mod[0]}, {c_vec_mod[1]}]</p>
        </div>
        """
    modulo_html += '</div>'
    html += modulo_html

    # Step 4: Vectors to Ciphertext
    html += '<h3 class="text-lg font-semibold text-green-400 mt-6 mb-2">4. Convert Back to Text</h3>'
    html += '<p class="text-gray-400 mb-4">Finally, the new numerical vectors are mapped back to characters to form the ciphertext.</p>'
    final_text_html = '<div class="flex flex-wrap gap-4">'
    for i, v in enumerate(result_vectors):
        chars = INT_TO_CHAR[v[0]] + INT_TO_CHAR[v[1]]
        final_text_html += f"""
        <div class="text-center p-3 bg-gray-800 rounded-lg">
            <p class="font-bold text-xl">[{v[0]}, {v[1]}]</p>
            <p class="text-sm text-gray-400">&#x2193;</p>
            <p class="font-mono">'{chars}'</p>
        </div>
        """
    final_text_html += '</div>'
    html += final_text_html
    html += f'<p class="mt-6 p-4 bg-green-900/50 border border-green-500 rounded-lg text-center font-mono text-xl tracking-widest">{encrypted}</p>'
    
    return flask.jsonify({'html': html})

# ==============================================================================
# 3. APPLICATION RUNNER
# ==============================================================================

def run_app():
    """Starts the Flask server."""
    # Use a non-standard port to avoid conflicts
    port = 5050
    # Open the web browser automatically
    url = f"http://127.0.0.1:{port}"
    # Use a timer to open the browser after the server has a moment to start
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    # Run the Flask app
    app.run(port=port, debug=False)

if __name__ == '__main__':
    print("Starting Matrix Encryption Demonstrator...")
    print("A new tab should open in your web browser shortly.")
    print("If not, please navigate to http://127.0.0.1:5050")
    run_app()
