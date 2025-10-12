# Matrix Encryption Demonstrator

A web-based visual tool that demonstrates the principles of the Hill Cipher, an encryption algorithm using invertible matrices and modular arithmetic. The application provides a hands-on experience with matrix-based cryptography in a clean, modern interface inspired by the Groww stock trading app.



## Overview

This project is a self-contained Python application that uses the Flask web framework to serve a single-page web app. It allows users to:
- See a live demonstration of text being encrypted and decrypted with a randomly generated key matrix.
- Encrypt and decrypt their own custom messages.
- View a detailed, step-by-step breakdown of the entire encryption process, from text-to-vector conversion to the final ciphertext generation.

The core cryptographic logic is implemented in Python using the NumPy library for efficient matrix operations.

---

## Features

- **Interactive Web UI**: A simple and intuitive interface built with Tailwind CSS.
- **Live Demonstration**: Generate random examples to see the encryption/decryption cycle in real-time.
- **User-Driven Encryption**: Encrypt your own plaintext or decrypt ciphertext using the session's key.
- **Detailed Visualizations**: A "Show Working" feature provides a clear, step-by-step explanation of the underlying mathematics.
- **Self-Contained**: The entire application (backend and frontend) is contained within a single Python script.
- **Automatic Browser Launch**: The script automatically opens the application in a new browser tab upon execution.

---

## Technology Stack

- **Backend**: Python, Flask
- **Numerical Operations**: NumPy
- **Frontend**: HTML, Tailwind CSS, Vanilla JavaScript

---

## Requirements

- Python 3.6+
- The Python packages listed in `requirements.txt`.

---

## Installation

1.  **Clone the repository or download the source code.**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To start the application, simply run the main Python script from your terminal:

```bash
python matrix_cipher_app.py
