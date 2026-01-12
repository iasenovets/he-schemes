import numpy as np
import random
from math import log2

def get_binary_vector(size):
    """Binary secret vector in {0,1}^k"""
    return np.random.randint(0, 2, size, dtype=np.int64)

def get_uniform_vector(size, q):
    """Uniform LWE a-vector in Z_q^k"""
    return np.random.randint(0, q, size, dtype=np.int64)

def get_normal_error(Δ):
    """Small integer error term (discrete Gaussian-ish)"""
    return random.randint(-Δ//4, Δ//4)  # Small noise, less than Δ/2 


def lwe_setup(q, t, k):
    """Setup LWE parameters"""
    if q % t != 0:
        raise ValueError("q must be divisible by t")
    
    Δ = q // t  # Scaling factor
    secret_key = get_binary_vector(k)  # Binary secret key s ∈ {0,1}^k
    
    print("=== LWE Setup ===")
    print(f"q (ciphertext modulus): {q}")
    print(f"t (plaintext modulus): {t}")
    print(f"k (vector length): {k}")
    print(f"Δ (scaling factor): {Δ} = q/t")
    print(f"Secret key s: {secret_key}")
    print(f"Plaintext range: [0, {t-1}]")
    print(f"Ciphertext range: [0, {q-1}]")
    print(f"Bits shifted by Δ: {int(log2(Δ))}")
    print()
    
    return q, t, k, Δ, secret_key

def lwe_encrypt(m, q, t, Δ, secret_key):
    """Encrypt a message m using LWE"""
    k = len(secret_key)
    
    # Step 1: Random public key vector
    a = get_uniform_vector(k, q) # Random vector a ∈ Z_q^k
    
    # Step 2: Small noise from Gaussian-like distribution
    # Using discrete Gaussian approximation (small values around 0)
    noise = get_normal_error(Δ) 
     
    # Step 3: Scale the message
    scaled_m = (Δ * m) % q
    
    # Step 4: Compute b = a·s + Δ·m + noise (mod q)
    dot_product = np.dot(a, secret_key) % q
    b = (dot_product + scaled_m + noise) % q
    
    print("=== LWE Encryption ===")
    print(f"Plaintext m: {m}")
    print(f"Scaled m (Δ·m): {scaled_m}")
    print(f"Random vector a: {a}")
    print(f"Secret key s: {secret_key}")
    print(f"a·s (mod q): {dot_product}")
    print(f"Noise e: {noise}")
    print(f"b = a·s + Δ·m + e (mod q): {b}")
    print(f"Ciphertext ct = (a, b)")
    print()
    
    return a, b, noise

def lwe_decrypt(a, b, q, t, Δ, secret_key):
    """Decrypt a ciphertext (a, b)"""
    # Step 1: Compute b - a·s (mod q)
    dot_product = np.dot(a, secret_key) % q
    decrypted_value = (b - dot_product) % q
    
    # Step 2: Round to nearest multiple of Δ
    # Since we're working with integers, we round to nearest multiple
    print(decrypted_value)
    rounded_value = round(decrypted_value / Δ) * Δ
    
    # Handle wrap-around at q boundary
    # Consider both rounded_value and rounded_value - q as candidates
    candidate1 = rounded_value
    candidate2 = rounded_value - q
    candidate3 = rounded_value + q
    
    # Choose the one closest to decrypted_value
    candidates = [candidate1, candidate2, candidate3]
    best_candidate = min(candidates, key=lambda x: abs(x - decrypted_value))
    
    # Step 3: Divide by Δ to recover m
    recovered_m = (best_candidate // Δ) % t
    
    print("=== LWE Decryption ===")
    print(f"Ciphertext (a, b): a={a}, b={b}")
    print(f"Compute b - a·s (mod q): {decrypted_value}")
    print(f"Value to round: {decrypted_value}")
    print(f"Rounded to nearest multiple of Δ={Δ}: {best_candidate}")
    print(f"Recovered m = rounded_value / Δ: {recovered_m}")
    print()
    
    return recovered_m, decrypted_value, best_candidate

def run_example(q, t, k, m):
    """Run a complete LWE example"""
    print("\n" + "="*50)
    print(f"EXAMPLE: q={q}, t={t}, k={k}, m={m}")
    print("="*50)
    
    # Setup
    q_val, t_val, k_val, Δ, secret_key = lwe_setup(q, t, k)
    
    # Encryption
    a, b, noise = lwe_encrypt(m, q_val, t_val, Δ, secret_key)
    
    # Decryption
    recovered_m, decrypted_value, rounded = lwe_decrypt(a, b, q_val, t_val, Δ, secret_key)
    
    # Verification
    print("=== Verification ===")
    print(f"Original m: {m}")
    print(f"Recovered m: {recovered_m}")
    print(f"Success: {m == recovered_m}")
    print(f"Noise magnitude: {abs(noise)}")
    print(f"Maximum allowed noise (Δ/2): {Δ//2}")
    print(f"Decryption works if |noise| < Δ/2: {abs(noise) < Δ//2}")
    
    return m == recovered_m

def demonstrate_noise_boundary(q, t, k, secret_key):
    """
    Show how decryption behaves as noise crosses the Δ/2 boundary.
    Uses a fixed a-vector and plaintext m to make the boundary visible.
    """
    Δ = q // t
    print("=== Noise Boundary Demonstration ===")
    print(f"q={q}, t={t}, Δ={Δ}, max safe noise < Δ/2 = {Δ//2}")
    print(f"Secret key s = {secret_key}")
    print()

    # Fix a small plaintext to see flipping
    m = 7
    scaled_m = (Δ * m) % q

    # Use a single fixed 'a' so that only noise changes
    a = [106, 71, 188, 20]
    dot_as = np.dot(a, secret_key) % q

    print(f"Using fixed plaintext m={m}")
    print(f"Using fixed a-vector: {a}")
    print(f"a·s mod q = {dot_as}")
    print()

    # Sweep noise values across the critical boundary
    test_noises = list(range(-(Δ), Δ + 1))

    for e in test_noises:
        b = (dot_as + scaled_m + e) % q
        decrypted_value = (b - dot_as) % q

        # Rounding step
        rounded = round(decrypted_value / Δ) * Δ
        recovered_m = (rounded // Δ) % t

        flag = "OK " if abs(e) < Δ/2 else "FAIL"
        print(f"noise={e:3d} | decrypted={decrypted_value:3d} | "
              f"rounded={rounded:3d} | m'={recovered_m} | {flag}")

    print("\nNotice where |noise| >= Δ/2 the plaintext jumps to the wrong value.")


def main():
    """Run multiple LWE examples"""
    # Example 1: Small parameters
    print("LWE (Learning With Errors) Implementation")
    print("Based on the provided mathematical description")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Example configurations
    examples = [
        # (q, t, k, m)
        (256, 16, 4, 7),    # q=256, t=16, Δ=16
        #(1024, 32, 8, 12),  # q=1024, t=32, Δ=32
        #(512, 8, 6, 3),     # q=512, t=8, Δ=64
    ]
    
    results = []
    for i, (q, t, k, m) in enumerate(examples, 1):
        success = run_example(q, t, k, m)
        results.append((i, success))
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for i, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"Example {i}: {status}")
    
    # Demonstrate noise boundary case
    print("\n" + "="*50)
    print("NOISE BOUNDARY DEMONSTRATION")
    print("="*50)
    print("The decryption works correctly when |noise| < Δ/2")
    print("If noise is too large, it corrupts the plaintext bits")
    print()
    
    q, t, k = 256, 16, 4
    Δ = q // t  # Δ = 16
    # Use a new secret key for this demonstration
    secret_key = get_binary_vector(k)

    demonstrate_noise_boundary(q, t, k, secret_key)

if __name__ == "__main__":
    main()