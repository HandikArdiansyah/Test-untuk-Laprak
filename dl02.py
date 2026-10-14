# ============================================================
# NEURON BELAJAR MEMBEDAKAN — 2 DATA (1 CACAT, 1 NORMAL)
# ============================================================

import math

print("="*65)
print("BAB 1 — SAYA PUNYA 2 PRODUK")
print("="*65)

# ---------- DATA ----------
# Produk A: keretakan tinggi, alur tinggi, warna tinggi
data = [
    {"x": [0.90, 0.80, 0.70], "label": 1, "nama": "Produk A"},  # CACAT
    {"x": [0.10, 0.20, 0.15], "label": 0, "nama": "Produk B"},  # NORMAL
]

for d in data:
    status = "CACAT" if d["label"] == 1 else "NORMAL"
    print(f"{d['nama']}: ciri={d['x']}, jawaban={d['label']} ({status})")

print("→ Tugas neuron: bedakan keduanya!\n")


print("="*65)
print("BAB 2 — SAYA ASAL TENTUKAN BOBOT & BIAS")
print("="*65)

bobot = [0.10, 0.20, -0.15]   # bobot awal (asal, jangan 0)
bias  = 0.05                   # bias awal (asal, jangan 0)

print(f"Bobot awal = {bobot}")
print(f"Bias awal  = {bias}\n")


print("="*65)
print("BAB 3 — PROSES BELAJAR (TEBAC → CEK → KOREKSI → ULANGI)")
print("="*65)

learning_rate = 0.5
epochs = 30

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


for epoch in range(1, epochs + 1):

    total_loss = 0
    total_koreksi_bobot = [0.0, 0.0, 0.0]
    total_koreksi_bias  = 0.0

    # ----- LOOP UNTUK SETIAP PRODUK -----
    for d in data:
        x = d["x"]
        y = d["label"]

        # A. TEBAK
        z = bobot[0]*x[0] + bobot[1]*x[1] + bobot[2]*x[2] + bias
        tebakan = sigmoid(z)

        # B. CEK SEBERAPA SALAH
        # BCE untuk 1 sampel
        loss = -(y*math.log(tebakan + 1e-9) +
                 (1-y)*math.log(1 - tebakan + 1e-9))
        total_loss += loss

        # C. HITUNG KOREKSI
        meleset = tebakan - y
        total_koreksi_bobot[0] += meleset * x[0]
        total_koreksi_bobot[1] += meleset * x[1]
        total_koreksi_bobot[2] += meleset * x[2]
        total_koreksi_bias      += meleset

    # D. RATA-RATA KOREKSI (dibagi jumlah data)
    n = len(data)
    total_koreksi_bobot[0] /= n
    total_koreksi_bobot[1] /= n
    total_koreksi_bobot[2] /= n
    total_koreksi_bias      /= n

    # E. TERAPKAN KOREKSI
    bobot[0] -= learning_rate * total_koreksi_bobot[0]
    bobot[1] -= learning_rate * total_koreksi_bobot[1]
    bobot[2] -= learning_rate * total_koreksi_bobot[2]
    bias     -= learning_rate * total_koreksi_bias

    # F. LAPORKAN
    avg_loss = total_loss / n
    print(f"Putaran {epoch:2d} | loss={avg_loss:.4f} | "
          f"bobot=[{bobot[0]:+.3f}, {bobot[1]:+.3f}, {bobot[2]:+.3f}] | "
          f"bias={bias:+.3f}")


print()
print("="*65)
print("BAB 4 — HASIL AKHIR: APAKAH NEURON SUDAH PINTAR MEMBEDAKAN?")
print("="*65)

print(f"Bobot akhir: {[round(b, 4) for b in bobot]}")
print(f"Bias akhir : {round(bias, 4)}\n")

# Uji ke kedua produk
for d in data:
    x = d["x"]
    z = bobot[0]*x[0] + bobot[1]*x[1] + bobot[2]*x[2] + bias
    tebakan = sigmoid(z)

    status_asli = "CACAT" if d["label"] == 1 else "NORMAL"
    status_tebak = "CACAT" if tebakan >= 0.5 else "NORMAL"
    benar = "✅ BENAR" if status_asli == status_tebak else "❌ SALAH"

    print(f"{d['nama']}: tebakan={tebakan:.4f} → {status_tebak} "
          f"(asli: {status_asli}) {benar}")