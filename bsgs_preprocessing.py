# Toy offline BSGS precompute demo for CPIR (server-side masked selection)
# Parameters: M=256 slots, r=16 slots/record, C_max=16, A=4, B=4
# This script shows how P_{j,a} plaintexts are constructed and how rotating the inner query aligns blocks.
# It prints detailed steps so you can follow the rotation and extraction visually.
import numpy as np
import textwrap

M = 256        # total slots
r = 16         # slots per record
C_max = 16     # records per shard
A = 4          # giant steps
B = 4          # baby steps per giant step
assert A * B >= C_max

def pretty_block(arr, start, length):
    return list(arr[start:start+length])

# Create toy shard records: for clarity, record u will contain the bytes [u, u, u, ...] length r
shard_records = []
for u in range(C_max):
    shard_records.append(np.full(r, u, dtype=np.uint8))

print("=== Toy shard records (u -> first 4 bytes shown) ===")
for u in range(C_max):
    print(f"u={u:2d}: bytes (first 8) = {shard_records[u][:8].tolist()}")

# OFFLINE: build P_{a} plaintext arrays for a single shard (P_j_a)
P = []  # list of plaintext arrays, one per a
for a in range(A):
    slots = np.zeros(M, dtype=np.int32)  # use int32 for printing clarity
    for b in range(B):
        u = a * B + b
        if u >= C_max:
            break
        slot_start = b * r
        # place record u into block b
        slots[slot_start:slot_start + r] = shard_records[u]
    P.append(slots)

print("\n=== Precomputed plaintexts P_{a} (blocks with nonzero values shown) ===")
for a, slots in enumerate(P):
    nonzero_blocks = []
    for b in range(B):
        slot_start = b * r
        block = slots[slot_start:slot_start + r]
        if np.any(block):
            nonzero_blocks.append((b, slot_start, block[:8].tolist()))
    print(f"P_{a}: nonzero blocks -> {nonzero_blocks}")

# Show full textual map of where each u was placed in P_{a}
print("\n=== Detailed placement: which P_a and which block b contains record u ===")
for u in range(C_max):
    a = u // B
    b = u % B
    print(f"u={u:2d} -> a={a}, b={b}, slot range in P_a: [{b*r} .. {b*r + r - 1}]")

# ONLINE: example query for u=9 (client-side)
u_target = 9
a_target = u_target // B
b_target = u_target % B
slot_start_target = u_target * r
print("\n=== Client target ===")
print(f"Target u = {u_target}  -> a={a_target}, b={b_target}, global slotStart = {slot_start_target}")

# Build inner query q_inner: one-hot over record u -> ones in slots [u*r .. u*r+r-1]
q_inner = np.zeros(M, dtype=np.int32)
q_inner[slot_start_target:slot_start_target + r] = 1

print("\n=== q_inner (nonzero slot range) ===")
print(f"ones at slots [{slot_start_target} .. {slot_start_target + r - 1}]")

# For each giant step a, rotate q_inner by -a*B*r slots and multiply with P[a].
parts = []
print("\n=== Per-giant-step rotation and partial products ===")
for a in range(A):
    rot_slots = (-a * B * r) % M  # numpy.roll uses positive shifts to the right, emulate
    q_rot = np.roll(q_inner, rot_slots)  # rotate right by rot_slots
    # For human clarity, compute the equivalent "logical rotation amount" printed as negative when small
    logical_rot = -a * B * r
    # find the nonzero slot ranges in rotated query
    nz = np.where(q_rot != 0)[0]
    nz_ranges = (nz[0], nz[-1]) if nz.size else (None, None)
    # Multiply (elementwise) rotated query with P[a] to get part
    part = q_rot * P[a]
    nonzero_indices = np.where(part != 0)[0]
    # For display, determine block b positions that became nonzero
    block_b_nonzero = sorted(list(set(idx // r for idx in nonzero_indices))) if nonzero_indices.size else []
    parts.append(part)
    print(f"\n a={a}: logical rot = {logical_rot:4d} slots -> numpy.roll shift = {rot_slots:3d}")
    print(f"   rotated q has ones at slots [{nz_ranges[0]} .. {nz_ranges[1]}] (if None => empty)")
    print(f"   P_{a} nonzero blocks (b indices) = {[b for (b,_,_) in [(bb,0,0) for bb in range(B) if np.any(P[a][bb*r:(bb+1)*r])]]}")
    print(f"   After multiply, nonzero block indices in part = {block_b_nonzero}")
    if block_b_nonzero:
        for b in block_b_nonzero:
            start = b * r
            print(f"     block b={b}, slots [{start}..{start+r-1}], sample values={part[start:start+8].tolist()}")

# Sum parts to get R_j
R_j = np.zeros(M, dtype=np.int32)
for part in parts:
    R_j += part

# Show the final R_j nonzero region
nz_final = np.where(R_j != 0)[0]
if nz_final.size:
    start_final, end_final = nz_final[0], nz_final[-1]
    # derive b_final where the result sits (client expects b*r .. b*r+r-1)
    b_final = start_final // r
    print("\n=== Final R_j (sum of parts) ===")
    print(f"Nonzero slots in R_j: [{start_final} .. {end_final}]  (should be r slots long)")
    print(f"Located in block b = {b_final}, slots [{b_final*r}..{b_final*r + r - 1}]")
    print(f"Sample of recovered record bytes: {R_j[b_final*r : b_final*r + r].tolist()}")
else:
    print("R_j is all zeros (unexpected)")

# Sanity: check that recovered record equals original shard_records[u_target]
recovered = R_j[b_target*r : b_target*r + r]
original = shard_records[u_target]
print("\n=== Sanity check ===")
print("Original record bytes (first 16):", original.tolist())
print("Recovered record bytes (first 16):", recovered.tolist())
print("Match:", np.array_equal(original, recovered))

# For completeness, show mapping of rotations that moved u_target block to canonical b positions
print("\n=== Rotation table: where u_target's block lands for each a ===")
for a in range(A):
    rot_slots = (-a * B * r) % M
    q_rot = np.roll(q_inner, rot_slots)
    nz = np.where(q_rot != 0)[0]
    if nz.size:
        print(f"a={a}: rot={-a*B*r:4d} slots  -> rotated ones at [{nz[0]}..{nz[-1]}]  -> aligned block index = {nz[0]//r}")
    else:
        print(f"a={a}: rot={-a*B*r:4d} slots  -> rotated ones empty")

