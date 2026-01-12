import numpy as np
from numpy.polynomial import polynomial as poly

# -----------------------
# Random polynomial gens
# -----------------------
def gen_binary_poly(size):
    return np.random.randint(0, 2, size, dtype=np.int64)

def gen_uniform_poly(size, modulus):
    return np.random.randint(0, modulus, size, dtype=np.int64)

def gen_normal_poly(size, mean, std):
    return np.int64(np.random.normal(mean, std, size=size)) # typical std 


# -----------------------
# Polynomial arithmetic in Z_q[X]/(f)
# -----------------------
def polymul(x, y, modulus, poly_mod, verbose=True):
    """Negacyclic (mod f) polynomial multiplication with coefficient mod q."""
    if verbose:
        print("\n[polymul]")
        print("   x(x):", poly_to_string(x, modulus))
        print("   y(x):", poly_to_string(y, modulus))

    raw_prod = poly.polymul(x, y)
    raw_mod = np.int64(np.round(raw_prod % modulus))
    quotient, remainder = poly.polydiv(raw_mod, poly_mod)
    remainder = np.int64(np.round(remainder % modulus))

    if verbose:
        print("   raw product:", poly_to_string(raw_prod, modulus))
        print("   remainder mod f(x):", poly_to_string(remainder, modulus))
        print("   result (mod q,f):", remainder, "\n")

    return remainder


def polyadd(x, y, modulus, poly_mod, verbose=True):
    """Addition of two polynomials reduced mod (q, f)."""
    if verbose:
        print("\n[polyadd]")
        print("   x(x):", poly_to_string(x, modulus))
        print("   y(x):", poly_to_string(y, modulus))

    raw_sum = poly.polyadd(x, y)
    raw_mod = np.int64(np.round(raw_sum % modulus))
    quotient, remainder = poly.polydiv(raw_mod, poly_mod)
    remainder = np.int64(np.round(remainder % modulus))

    if verbose:
        print("   raw sum:", poly_to_string(raw_sum, modulus))
        print("   remainder mod f(x):", poly_to_string(remainder, modulus))
        print("   result (mod q,f):", remainder, "\n")

    return remainder


# -----------------------
# RLWE Setup
# -----------------------
def rlwe_setup(size, q, t, poly_mod):
    """
    Return RLWE parameters and secret key.
    Samples:
        - Δ = q/t
        - secret key s(x)
        - random A(x)
    """
    Delta = q // t
    s = gen_binary_poly(size)
    A = gen_uniform_poly(size, q)

    print("[+] RLWE setup:")
    print("    secret s(x):", poly_to_string(s, q))
    print("    A(x):", poly_to_string(A, q))
    print("    q =", q, ", t =", t, ", Δ =", Delta)

    return q, t, Delta, A, s


# -----------------------
# RLWE Encrypt: (A, B) with B = A*S + Δ*M + E
# -----------------------
def rlwe_encrypt(size, q, t, Delta, A, s, poly_mod, m_int, std):
    """
    Encrypt pt m under RLWE:
        A is public
        B = A*s + Δ*m + e
    Ciphertext = (A, B)
    """
    # encode plaintext into polynomial
    M = np.array([m_int] + [0] * (size - 1), dtype=np.int64) % t
    scaled_M = (Delta * M) % q
    E = gen_normal_poly(size, 0, std)  # small error polynomial
    print(E)

    print("\n[+] RLWE encrypt")
    print("    m(x):", poly_to_string(M, t))
    print("    Δ*m(x):", poly_to_string(scaled_M, q))
    print("    e(x):", poly_to_string(E, q))

    As = polymul(A, s, q, poly_mod, verbose=True)
    B_raw = polyadd(As, scaled_M, q, poly_mod, verbose=True)
    B = polyadd(B_raw, E, q, poly_mod, verbose=True)

    return (A, B)


# -----------------------
# RLWE Decrypt
# -----------------------
def rlwe_decrypt(size, q, t, Delta, s, poly_mod, ct):
    """
    Decrypt:
        ct = (A, B)
        Compute: B - A*s = Δ*M + E
        Then divide by Δ and round mod t
    """
    A, B = ct

    print("\n[+] RLWE decrypt")

    As = polymul(A, s, q, poly_mod, verbose=True)
    diff = polyadd(B, -As, q, poly_mod, verbose=True)

    print("    (B - A*s)(x):", poly_to_string(diff, q))

    recovered_poly = np.round(diff / Delta) % t
    recovered = int(recovered_poly[0])

    print("    recovered polynomial:", recovered_poly)
    print("    recovered message:", recovered)

    return recovered


# -----------------------
# Utility
# -----------------------
def poly_to_string(coeffs, modulus=None):
    terms = []
    for i, c in enumerate(coeffs):
        if modulus is not None:
            c = int(c % modulus)
        if c == 0:
            continue
        if i == 0:
            terms.append(f"{c}")
        elif i == 1:
            terms.append(f"{c}*x")
        else:
            terms.append(f"{c}*x^{i}")
    return " + ".join(terms) if terms else "0"


# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    n = 4
    q = 256
    t = 16
    Δ = q // t
    poly_mod = np.array([1] + [0]*(n-1) + [1])  # x^n + 1
    std = Δ // 4

    print("=== RLWE Demo ===")

    q, t, Delta, A, s = rlwe_setup(n, q, t, poly_mod)

    m = 3
    ct = rlwe_encrypt(n, q, t, Delta, A, s, poly_mod, m, std)

    print("\nCiphertext A:", ct[0])
    print("Ciphertext B:", ct[1])

    recovered = rlwe_decrypt(n, q, t, Delta, s, poly_mod, ct)
    print("\nRecovered:", recovered)
