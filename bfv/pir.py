import numpy as np
from numpy.polynomial import polynomial as poly
import bfv_2 as rlwe_updated

def print_visuals(db_matrix, selection_vector, target_index, n, q):
    print("\n" + "="*60)
    print("             LINEAR ALGEBRA VISUALIZATION")
    print("="*60)
    
    print("\n[DATABASE - D (Server Side)]")
    print(f"Dimensions: {len(db_matrix)} × {n}")
    print("Format: D[i] = polynomial of degree {n-1} (coefficients vector)")
    print("-"*40)
    
    # Show database structure
    for i in range(min(4, len(db_matrix))):  # Show first 4 rows
        row_label = f"D[{i}] (row {i})"
        coeffs_str = str(db_matrix[i][:5]) + "..." if n > 5 else str(db_matrix[i])
        if i == target_index:
            print(f"{row_label:12} = {coeffs_str}  <-- TARGET RECORD")
        else:
            print(f"{row_label:12} = {coeffs_str}")
    
    if len(db_matrix) > 4:
        print(f"... and {len(db_matrix)-4} more rows")
    
    print("\n[SELECTION VECTOR - s (Client Encrypted)]")
    print(f"Dimensions: {len(selection_vector)} × 1 (each element is a ciphertext)")
    print("Format: s[i] = Encrypt(1 if i=target else 0)")
    print("-"*40)
    
    # Show selection vector structure
    for i in range(min(4, len(selection_vector))):
        ct0_len = len(selection_vector[i][0])
        ct1_len = len(selection_vector[i][1])
        if i == target_index:
            print(f"s[{i}] = Enc(1)  --> CT dimensions: ({ct0_len},) × ({ct1_len},)")
        else:
            print(f"s[{i}] = Enc(0)  --> CT dimensions: ({ct0_len},) × ({ct1_len},)")
    
    if len(selection_vector) > 4:
        print(f"... and {len(selection_vector)-4} more encrypted elements")
    
    print("\n[OPERATION: Dot Product]")
    print("result = Σ_i (s[i] ⊗ D[i])")
    print("where: ⊗ = homomorphic multiplication")
    print("       Σ = homomorphic addition")
    
    print("\n[MATH BREAKDOWN]")
    print("result = s[0]·D[0] + s[1]·D[1] + ... + s[m-1]·D[m-1]")
    print(f"        = Enc(0)·D[0] + Enc(0)·D[1] + ... + Enc(1)·D[{target_index}] + ...")
    print(f"        ≈ Enc(0 + 0 + ... + D[{target_index}] + ...)")
    print(f"        = Enc(D[{target_index}])")
    
    print("\n[COMPUTATIONAL COMPLEXITY]")
    print("\n[COMPUTATIONAL COMPLEXITY - FULL BREAKDOWN]")
    print("For each database record i (0 ≤ i < 4):")
    print("  Step 1: Multiply s[i] × D[i] (plaintext-ciphertext multiplication)")
    print("          - c0_i × D[i]  (polynomial multiplication mod x^n+1)")
    print("          - c1_i × D[i]  (polynomial multiplication mod x^n+1)")
    print("")
    print("  Step 2: Add to accumulator")
    print("          - accum_c0 += (c0_i × D[i])")
    print("          - accum_c1 += (c1_i × D[i])")
    print("")
    print(f"Total operations for {len(selection_vector)} records:")
    print(f"  Polynomial multiplications: {2 * len(selection_vector)}")
    print(f"  Polynomial additions: {2 * (len(selection_vector) + 1)}  (including initialization)")
    print(f"  All operations modulo q = {q} and polynomial x^{n} + 1")
    print("="*60 + "\n")


def run_pir_example():
    # --- 1. Parameters Setup ---
    n = 2 ** 10         # 64 coefficients
    q = 2 ** 29        # Ciphertext modulus 4096
    t = 2 ** 8         # Plaintext modulus 256
    std = 1            # Standard deviation for error
    poly_mod = np.array([1] + [0] * (n - 1) + [1]) # x^n + 1

    print(f"--- Setting up PIR with n={n}, q={q}, t={t} ---")
    
    # Generate Keys (Client Side)
    print("[1/5] Key Generation (Client)...")
    pk, sk = rlwe_updated.keygen(n, q, poly_mod, std)
    print("Keys generated.")

    # --- 2. Create the Database (Server Side) ---
    # We simulate 'x' records. 
    # To respect the tight noise budget of q=4096, we use small integers.
    # In a real scenario with larger q, these could be ASCII codes.
    
    # Let's say we have 4 records (rows)
    # Each record is a polynomial of degree n
    print("[2/5] Database Creation (Server)...")
    db_size_x = 4
    
    # Creating dummy data (Rows of the matrix)
    # Row 0: All 1s
    # Row 1: All 2s
    # Row 2: Pattern [1, 2, 3, 4...] (This is our TARGET)
    # Row 3: All 5s
    
    db_matrix = []
    db_matrix.append([1] * n)                 # Record 0
    db_matrix.append([2] * n)                 # Record 1
    db_matrix.append([(i % 10) for i in range(n)]) # Record 2 (Target)
    db_matrix.append([5] * n)                 # Record 3
    
    print(f"Database created with {len(db_matrix)} records.")
    print()

    # --- 3. Client Generates Query ---
    print("[3/5] Query Generation (Client)...")
    target_index = 2
    print(f"Client wants to retrieve record at index: {target_index}")

    # Create Selection Vector: [Enc(0), Enc(0), Enc(1), Enc(0)]
    selection_vector = []
    for i in range(db_size_x):
        if i == target_index:
            # Encrypt 1
            ct = rlwe_updated.encrypt(pk, n, q, t, poly_mod, [1], std)
        else:
            # Encrypt 0
            ct = rlwe_updated.encrypt(pk, n, q, t, poly_mod, [0], std)
        selection_vector.append(ct)
    
    print(f"Query (Selection vector created: {len(selection_vector)}) encrypted elements and sent to server.")

    # --- VISUALIZE BEFORE PROCESSING ---
    print_visuals(db_matrix, selection_vector, target_index, n, q)

    # --- 4. Server Processes Query (Homomorphic Matrix Multiplication) ---
    print("[4/5] Server Processing (Homomorphic Computation)...")
    
    # We initialize the accumulator with an encryption of zero (or a "zero ciphertext")
    # A zero ciphertext in BFV is just (0, 0) polynomials
    accum_ct0 = np.zeros(n, dtype=np.int64)
    accum_ct1 = np.zeros(n, dtype=np.int64)
    accumulator = (accum_ct0, accum_ct1)

    print(f"Server computing dot product: result = Σ_i selection[i] × database[i]")
    
    for i in range(db_size_x):
        # Step A: Multiply Selection[i] * Record[i]
        # We use mul_plain because Selection is Ciphertext and Record is Plaintext
        product = rlwe_updated.mul_plain(selection_vector[i], db_matrix[i], q, t, poly_mod)
        # Step B: Add to accumulator
        accumulator = rlwe_updated.add_cipher(accumulator, product, q, poly_mod)
        # Show progress
        if i < 3 or i == db_size_x - 1:
            op = "×" if i == target_index else "·0"
            print(f"   + Enc({1 if i==target_index else 0}) {op} D[{i}]")

    print("Computation complete. Result sent to client.")
    print()

    # --- 5. Client Decrypts Result ---
    print("[5/5] Result Decryption (Client)...")
    decrypted_result = rlwe_updated.decrypt(sk, n, q, t, poly_mod, accumulator)
    expected_result = db_matrix[target_index]

    # Output verification
    print("\n--- Results ---")
    print(f"Decrypted Result (First 10 coeffs): {decrypted_result[:10]}")
    print(f"Expected Result  (First 10 coeffs): {db_matrix[target_index][:10]}")
    print(f"  Database D ∈ (Z_t)^{{{db_size_x}×{n}}}")
    print(f"  Selection s ∈ Enc({{0,1}})^{db_size_x}")
    print(f"  Result = s·D = Enc(D[target])")

    
    if np.array_equal(decrypted_result, db_matrix[target_index]):
        print("SUCCESS: Retrieved the correct record!")
    else:
        print("FAILURE: Decryption did not match. (Likely noise overflow due to small q)")


if __name__ == "__main__":
    run_pir_example()