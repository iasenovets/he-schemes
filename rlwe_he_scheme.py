"""A basic homomorphic encryption scheme inspired from BFV https://eprint.iacr.org/2012/144.pdf
You can read my blog post explaining the implementation details here: https://www.ayoub-benaissa.com/blog/build-he-scheme-from-scratch-python/
Disclaimer: This implementation doesn’t neither claim to be secure nor does it follow software engineering best practices,
it is designed as simple as possible for the reader to understand the concepts behind homomorphic encryption schemes.
"""

import numpy as np
from numpy.polynomial import polynomial as poly

# Functions for random polynomial generation


def gen_binary_poly(size):
    """Generates a polynomial with coeffecients in [0, 1]
    Args:
        size: number of coeffcients, size-1 being the degree of the
            polynomial.
    Returns:
        array of coefficients with the coeff[i] being 
        the coeff of x ^ i.
    """
    return np.random.randint(0, 2, size, dtype=np.int64)


def gen_uniform_poly(size, modulus):
    """Generates a polynomial with coeffecients being integers in Z_modulus
    Args:
        size: number of coeffcients, size-1 being the degree of the
            polynomial.
    Returns:
        array of coefficients with the coeff[i] being 
        the coeff of x ^ i.
    """
    return np.random.randint(0, modulus, size, dtype=np.int64)


def gen_normal_poly(size, mean, std):
    """Generates a polynomial with coeffecients in a normal distribution
    of mean 0 and a standard deviation of 2, then discretize it.
    Args:
        size: number of coeffcients, size-1 being the degree of the
            polynomial.
        mean: mean of the normal distribution.
        std: standard deviation of the normal distribution.
    Returns:
        array of coefficients with the coeff[i] being 
        the coeff of x ^ i.
    """
    return np.int64(np.random.normal(mean, std, size=size))


# Functions for polynomial evaluation in Z_q[X]/(X^N + 1)

def polymul(x, y, modulus, poly_mod, verbose=True):
    """Multiply two polynomials and print intermediate steps."""
    if verbose:
        print("\n[polymul]")
        print("   x(x):", poly_to_string(x, modulus))
        print("   y(x):", poly_to_string(y, modulus))

    # raw product (no modular reduction yet)
    raw_prod = poly.polymul(x, y)
    if verbose:
        print("   raw product:", poly_to_string(raw_prod, modulus))

    # polynomial division to reduce mod poly_mod
    quotient, remainder = poly.polydiv(raw_prod % modulus, poly_mod)
    remainder = np.int64(np.round(remainder % modulus))

    if verbose:
        print("   remainder mod f(x):", poly_to_string(remainder, modulus))
        print("   result (mod q,f):", remainder, "\n")

    return remainder


def polyadd(x, y, modulus, poly_mod, verbose=True):
    """Add two polynomials and print intermediate steps."""
    if verbose:
        print("\n[polyadd]")
        print("   x(x):", poly_to_string(x, modulus))
        print("   y(x):", poly_to_string(y, modulus))

    raw_sum = poly.polyadd(x, y)
    if verbose:
        print("   raw sum:", poly_to_string(raw_sum, modulus))

    quotient, remainder = poly.polydiv(raw_sum % modulus, poly_mod)
    remainder = np.int64(np.round(remainder % modulus))

    if verbose:
        print("   remainder mod f(x):", poly_to_string(remainder, modulus))
        print("   result (mod q,f):", remainder, "\n")

    return remainder


# Functions for keygen, encryption and decryption
def keygen(size, modulus, poly_mod, std):
    """Generate a public and secret keys
    Args:
        size: size of the polynoms for the public and secret keys.
        modulus: coefficient modulus.
        poly_mod: polynomial modulus.
        std: standard deviation for the error polynomial.
    Returns:
        Public and secret key.
    """
    s = gen_binary_poly(size)
    a = gen_uniform_poly(size, modulus)
    e = gen_normal_poly(size, 0, std)
    b = polyadd(polymul(-a, s, modulus, poly_mod), -e, modulus, poly_mod)

    print("[+] Generated secret key s(x):", poly_to_string(s, modulus))
    print("[+] Generated public key pk(x):", poly_to_string(b, modulus), ",", poly_to_string(a, modulus))
    print(f"[+] Generated error polynomial e(x):", poly_to_string(e, modulus))

    return (b, a), s


def encrypt(pk, size, q, t, poly_mod, m, std):
    """Encrypt an integer.
    Args:
        pk: public-key (pk[0], pk[1]).
        size: size of polynomials. (degree = size - 1)
        q: ciphertext modulus. (q >> t)
        t: plaintext modulus. (t < q)
        poly_mod: polynomial modulus. (e.g., x^n + 1)
        m: integer to be encrypted.
        std: standard deviation for the error polynomials.
    Returns:
        Tuple representing a ciphertext.      
    """
    # encode the integer into a plaintext polynomial
    m = np.array([m] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m
    e1 = gen_normal_poly(size, 0, std)
    e2 = gen_normal_poly(size, 0, std)
    u = gen_binary_poly(size)
    print("[+] Plaintext polynomial m(x):", poly_to_string(m, t))
    print("[+] Delta value:", delta)
    print("[+] Scaled plaintext polynomial Δ*m(x):", poly_to_string(scaled_m, q))
    print("[+] Generated error polynomial e1(x):", poly_to_string(e1, q))
    print("[+] Generated error polynomial e2(x):", poly_to_string(e2, q))
    print("[+] Generated random polynomial u(x):", poly_to_string(u, q))

    ct0 = polyadd(
        polyadd(
            polymul(pk[0], u, q, poly_mod),
            e1, q, poly_mod),
        scaled_m, q, poly_mod
    )
    ct1 = polyadd(
        polymul(pk[1], u, q, poly_mod),
        e2, q, poly_mod
    )

    return (ct0, ct1)


def decrypt(sk, size, q, t, poly_mod, ct):
    """Decrypt a ciphertext
    Args:
        sk: secret-key.
        size: size of polynomials.
        q: ciphertext modulus.
        t: plaintext modulus.
        poly_mod: polynomial modulus.
        ct: ciphertext.
    Returns:
        Integer representing the plaintext.
    """
    scaled_pt = polyadd(
        polymul(ct[1], sk, q, poly_mod),
        ct[0], q, poly_mod
    )
    delta = q // t
    decrypted_poly = np.round(scaled_pt / delta) % t
    return int(decrypted_poly[0])


# Function for adding and multiplying encrypted values

def add_plain(ct, pt, q, t, poly_mod):
    """Add a ciphertext and a plaintext.
    Args:
        ct: ciphertext.
        pt: integer to add.
        q: ciphertext modulus.
        t: plaintext modulus.
        poly_mod: polynomial modulus.
    Returns:
        Tuple representing a ciphertext.
    """
    size = len(poly_mod) - 1
    # encode the integer into a plaintext polynomial
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    delta = q // t
    scaled_m = delta * m
    new_ct0 = polyadd(ct[0], scaled_m, q, poly_mod)
    return (new_ct0, ct[1])


def add_cipher(ct1, ct2, q, poly_mod):
    """Add a ciphertext and a ciphertext.
    Args:
        ct1, ct2: ciphertexts.
        q: ciphertext modulus.
        poly_mod: polynomial modulus.
    Returns:
        Tuple representing a ciphertext.
    """
    new_ct0 = polyadd(ct1[0], ct2[0], q, poly_mod)
    new_ct1 = polyadd(ct1[1], ct2[1], q, poly_mod)
    return (new_ct0, new_ct1)


def mul_plain(ct, pt, q, t, poly_mod):
    """Multiply a ciphertext and a plaintext.
    Args:
        ct: ciphertext.
        pt: integer to multiply.
        q: ciphertext modulus.
        t: plaintext modulus.
        poly_mod: polynomial modulus.
    Returns:
        Tuple representing a ciphertext.
    """
    size = len(poly_mod) - 1
    # encode the integer into a plaintext polynomial
    m = np.array([pt] + [0] * (size - 1), dtype=np.int64) % t
    new_c0 = polymul(ct[0], m, q, poly_mod)
    new_c1 = polymul(ct[1], m, q, poly_mod)
    return (new_c0, new_c1)

def poly_to_string(coeffs, modulus=None):
    """
    Convert a coefficient array into a human-readable polynomial string.
    coeffs[i] = coefficient of x^i.

    Args:
        coeffs: numpy array of coefficients
        modulus: optional modulus, to reduce coefficients

    Returns:
        str representation of the polynomial
    """
    terms = []
    for i, c in enumerate(coeffs):
        if modulus:
            c = c % modulus
        if c == 0:
            continue
        if i == 0:
            terms.append(f"{c}")
        elif i == 1:
            terms.append(f"{c}*x")
        else:
            terms.append(f"{c}*x^{i}")
    return " + ".join(terms) if terms else "0"



if __name__ == "__main__":
    # Scheme's parameters
    # polynomial modulus degree
    n = 2 ** 2 
    # ciphertext modulus
    q = 2 ** 5 #32
    # plaintext modulus
    t = 2
    # polynomial modulus
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    # standard deviation for error polynomials
    std = 1

    print("=== Tiny BGV Example ===")
    print(f"Working with polynomials of degree < {n}")
    print(f"Plaintext space: coefficients in Z_{t}")
    print(f"Ciphertext space: coefficients in Z_{q}")
    print(f"Polynomial modulus: x^{n} + 1")

    # Keygen
    print(f"[+] Key Generation")
    pk, sk = keygen(n, q, poly_mod, std)
    print(f"public key:{pk}")
    print(f"secret key:{sk}")

    # Encryption
    print(f"[+] Encryption")
    pt1, pt2 = 73, 20
    cst1, cst2 = 7, 5
    print(f"plaintexts: pt1 = {pt1}, pt2 = {pt2}")
    print(f"constants: cst1 = {cst1}, cst2 = {cst2}")

    ct1 = encrypt(pk, n, q, t, poly_mod, pt1, std)
    ct2 = encrypt(pk, n, q, t, poly_mod, pt2, std)
    print(f"ciphertexts: ct1 = {ct1}, ct2 = {ct2}")

    print("[+] Ciphertext ct1({}):".format(pt1))
    print("")
    print("\t ct1_0:", ct1[0])
    print("\t ct1_1:", ct1[1])
    print("")
    print("[+] Ciphertext ct2({}):".format(pt2))
    print("")
    print("\t ct2_0:", ct2[0])
    print("\t ct2_1:", ct2[1])
    print("")

    # Evaluation
    ct3 = add_plain(ct1, cst1, q, t, poly_mod)
    ct4 = mul_plain(ct2, cst2, q, t, poly_mod)
    # ct5 = ct1 + 7 + 3 * ct2
    ct5 = add_cipher(ct3, ct4, q, poly_mod)

    # Decryption
    decrypted_ct3 = decrypt(sk, n, q, t, poly_mod, ct3)
    decrypted_ct4 = decrypt(sk, n, q, t, poly_mod, ct4)
    decrypted_ct5 = decrypt(sk, n, q, t, poly_mod, ct5)
    print("[+] Decrypted ct3(ct1 + {}): {}".format(cst1, decrypted_ct3))
    print("[+] Decrypted ct4(ct2 * {}): {}".format(cst2, decrypted_ct4))
    print("[+] Decrypted ct5(ct1 + {} + {} * ct2): {}".format(cst1, cst2, decrypted_ct5))