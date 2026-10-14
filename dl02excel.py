# pip install openpyxl pandas


import math
import pandas as pd

# ---------- 1. Baca data dari Excel ----------
df = pd.read_excel("data.xlsx")

fitur_kolom = ["keretakan", "ketidakrataan_alur", "beda_warna"]
X = df[fitur_kolom].values.tolist()
Y = df["label"].values.tolist()

print(f"Jumlah data terbaca: {len(X)} baris")

# ---------- 2. Inisialisasi bobot & bias ----------
w = [0.10, 0.20, -0.15]
b = 0.05
learning_rate = 0.3
epochs = 100

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))

# ---------- 3. Training loop ----------
for epoch in range(epochs):
    total_loss = 0.0

    for xi_row, yi_label in zip(X, Y):
        # ----- forward -----
        z = sum(xi * wi for xi, wi in zip(xi_row, w)) + b
        y_pred = sigmoid(z)

        # ----- loss (MSE) -----
        total_loss += (y_pred - yi_label) ** 2

        # ----- backward (chain rule) -----
        d_loss_d_z = 2 * (y_pred - yi_label) * y_pred * (1 - y_pred)
        gradien_w = [d_loss_d_z * xi for xi in xi_row]
        gradien_b = d_loss_d_z

        # ----- update (SGD: update per sampel) -----
        w = [wi - learning_rate * gi for wi, gi in zip(w, gradien_w)]
        b = b - learning_rate * gradien_b

    rata_loss = total_loss / len(X)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | rata-rata loss: {rata_loss:.4f}")

print("\nBobot akhir:", w)
print("Bias akhir :", b)

# ---------- 4. Uji prediksi sampel baru ----------
def prediksi(fitur):
    z = sum(f * wi for f, wi in zip(fitur, w)) + b
    return sigmoid(z)

contoh_baru = [0.80, 0.60, 0.50]
hasil = prediksi(contoh_baru)
status = "CACAT" if hasil >= 0.5 else "NORMAL"
print(f"\nPrediksi ban baru: {hasil:.4f} -> {status}")