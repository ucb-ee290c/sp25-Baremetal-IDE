import numpy as np

def emit_case(name, M, N, K, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-8, 8, size=(M,K), dtype=np.int8)
    B = rng.integers(-8, 8, size=(K,N), dtype=np.int8)
    print(f"/* ---------- {name} ({M}x{N}x{K}) ---------- */")
    print(f"static const int8_t A_{name}[{M}*{K}] = {{")
    rows = [",".join(f"{int(x):3d}" for x in row) for row in A]
    for r in rows: print("" + r + ",")
    print("};")
    print(f"static const int8_t B_{name}[{K}*{N}] = {{")
    rows = [",".join(f"{int(x):3d}" for x in row) for row in B]
    for r in rows: print("" + r + ",")
    print("};")
    print(f'{{ "{name}", {M}, {N}, {K}, A_{name}, B_{name} }},')
    print()

# emit_case("aligned_128x128x128", 128, 128, 128, seed=3)
# emit_case("aligned_64x64x64", 64, 64, 64, seed=1)
# emit_case("unaligned_13x11x29", 13, 11, 29, seed=2)

emit_case("aligned_16x16x16", 16, 16, 16, seed=2)
