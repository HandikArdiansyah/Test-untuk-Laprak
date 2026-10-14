# Jalankan:  pip install -r requirements.txt
#            streamlit run app.py
import io
import math
import random
from datetime import datetime

import pandas as pd
import streamlit as st

BIRU, HIJAU = "#60a5fa", "#14b8a6"
FITUR = ["keretakan", "ketidakrataan_alur", "beda_warna"]
LABEL = "label"

st.set_page_config(page_title="Deteksi Cacat Ban", layout="centered")

st.markdown(f"""
<style>
.stApp {{ background:#f8fafc; }}
h1,h2,h3 {{ color:#0f172a; letter-spacing:-.01em; }}
.aksen {{ height:4px; width:64px; border-radius:4px;
        background:linear-gradient(90deg,{BIRU},{HIJAU}); margin:-6px 0 18px; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid #e2e8f0;
        border-radius:12px; padding:12px 16px; }}
.stButton>button, .stDownloadButton>button {{ border-radius:10px; border:1px solid {BIRU};
        background:{BIRU}; color:#fff; font-weight:600; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:{HIJAU};
        background:{HIJAU}; color:#fff; }}
.kartu {{ background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:18px 20px; }}
.badge {{ display:inline-block; padding:4px 14px; border-radius:999px; color:#fff; font-weight:700; }}
button[role="tab"][aria-selected="true"] {{ color:{BIRU}; }}
</style>
""", unsafe_allow_html=True)


# ---------------- Neuron tunggal (sama seperti dl02excel.py) ----------------
def sigmoid(z):
    z = max(-500, min(500, z))
    return 1 / (1 + math.exp(-z))


def latih(X, Y, w, b, lr, epochs, acak, seed):
    riwayat = []
    idx = list(range(len(X)))
    rng = random.Random(seed)
    for ep in range(1, epochs + 1):
        if acak:
            rng.shuffle(idx)
        total_loss, benar = 0.0, 0
        for i in idx:
            xi, yi = X[i], Y[i]
            z = sum(a * c for a, c in zip(xi, w)) + b
            y_pred = sigmoid(z)
            total_loss += (y_pred - yi) ** 2
            benar += int((y_pred >= 0.5) == (yi == 1))
            d = 2 * (y_pred - yi) * y_pred * (1 - y_pred)
            w = [c - lr * d * a for c, a in zip(w, xi)]
            b = b - lr * d
        riwayat.append({"epoch": ep, "loss": total_loss / len(X), "akurasi": benar / len(X)})
    return w, b, riwayat


def skor(fitur, w, b):
    return sigmoid(sum(f * c for f, c in zip(fitur, w)) + b)


def prediksi_df(df, w, b, ambang):
    hasil = df.copy()
    hasil["probabilitas_cacat"] = [round(skor(r, w, b), 4) for r in df[FITUR].values.tolist()]
    hasil["prediksi"] = ["CACAT" if p >= ambang else "NORMAL" for p in hasil["probabilitas_cacat"]]
    if LABEL in hasil.columns:
        hasil["aktual"] = hasil[LABEL].map({1: "CACAT", 0: "NORMAL"})
        hasil["benar"] = hasil["prediksi"] == hasil["aktual"]
    return hasil


def baca_excel(file):
    df = pd.read_excel(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def cek_kolom(df, butuh_label):
    wajib = FITUR + ([LABEL] if butuh_label else [])
    hilang = [c for c in wajib if c not in df.columns]
    if hilang:
        st.error(f"Kolom tidak ditemukan: {', '.join(hilang)}. Kolom wajib: {', '.join(wajib)}.")
        return False
    if df[wajib].isnull().any().any():
        st.error("Ada sel kosong pada kolom wajib. Lengkapi dulu datanya.")
        return False
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in wajib):
        st.error("Semua kolom fitur/label harus berupa angka.")
        return False
    if butuh_label and not set(df[LABEL].unique()) <= {0, 1}:
        st.error("Kolom label hanya boleh berisi 0 (normal) atau 1 (cacat).")
        return False
    return True


def excel_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for nama, d in sheets.items():
            d.to_excel(xw, sheet_name=nama, index=False)
    return buf.getvalue()


def data_contoh(n=40):
    r = random.Random(7)
    baris = []
    for _ in range(n):
        cacat = r.random() < 0.5
        lo, hi = (0.55, 1.0) if cacat else (0.0, 0.45)
        baris.append({
            "keretakan": round(r.uniform(lo, hi), 2),
            "ketidakrataan_alur": round(r.uniform(lo, hi), 2),
            "beda_warna": round(r.uniform(max(0, lo - .1), min(1, hi + .1)), 2),
            "label": int(cacat),
        })
    return pd.DataFrame(baris)


def laporan_html(ss, hasil):
    p = ss["param"]
    riw = ss["riwayat"]
    n = len(hasil)
    n_cacat = int((hasil["prediksi"] == "CACAT").sum())
    # grafik loss inline SVG
    W, H = 560, 180
    mx = max(r["loss"] for r in riw) or 1
    pts = " ".join(f"{20 + i * (W - 40) / max(1, len(riw) - 1):.1f},{H - 20 - r['loss'] / mx * (H - 40):.1f}"
                for i, r in enumerate(riw))
    svg = (f"<svg viewBox='0 0 {W} {H}' width='100%'><polyline fill='none' stroke='{BIRU}' "
            f"stroke-width='2.5' points='{pts}'/><line x1='20' y1='{H-20}' x2='{W-20}' y2='{H-20}' stroke='#cbd5e1'/></svg>")
    akurasi = ""
    if "benar" in hasil.columns:
        akurasi = f"<div class='k'><b>{hasil['benar'].mean():.1%}</b><span>Akurasi</span></div>"
    baris = "".join(
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in hasil.astype(str).values.tolist())
    kepala = "".join(f"<th>{c}</th>" for c in hasil.columns)
    return f"""<!doctype html><html lang="id"><meta charset="utf-8"><title>Laporan Deteksi Cacat Ban</title>
<style>body{{font-family:system-ui,sans-serif;background:#f8fafc;color:#0f172a;max-width:860px;margin:32px auto;padding:0 16px}}
h1{{margin-bottom:4px}} .bar{{height:4px;width:64px;border-radius:4px;background:linear-gradient(90deg,{BIRU},{HIJAU});margin-bottom:20px}}
.row{{display:flex;gap:12px;flex-wrap:wrap}} .k{{flex:1;min-width:140px;background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px 16px}}
.k b{{display:block;font-size:22px;color:{HIJAU}}} .k span{{color:#64748b;font-size:13px}}
.c{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin:16px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th{{background:{BIRU};color:#fff;text-align:left}}
th,td{{padding:6px 10px;border-bottom:1px solid #e2e8f0}}</style>
<h1>Laporan Deteksi Cacat Ban</h1><div class="bar"></div>
<p>Dibuat: {datetime.now():%d %B %Y, %H:%M}</p>
<div class="row"><div class="k"><b>{n}</b><span>Ban diperiksa</span></div>
<div class="k"><b>{n_cacat}</b><span>Terdeteksi cacat</span></div>
<div class="k"><b>{n - n_cacat}</b><span>Normal</span></div>{akurasi}</div>
<div class="c"><h3>Parameter model</h3>
<p>Learning rate: {p['lr']} · Epoch: {p['epochs']} · Ambang: {p['ambang']} · Data latih: {p['n_latih']} baris</p>
<p>Bobot akhir: {[round(x, 4) for x in ss['w']]} · Bias akhir: {round(ss['b'], 4)}</p>
<p>Loss awal {riw[0]['loss']:.4f} → loss akhir {riw[-1]['loss']:.4f}</p>{svg}</div>
<div class="c"><h3>Hasil prediksi</h3><table><tr>{kepala}</tr>{baris}</table></div></html>"""


# ---------------- UI ----------------
ss = st.session_state
st.title("Deteksi Cacat Ban")
st.markdown('<div class="aksen"></div>', unsafe_allow_html=True)
st.caption("Neuron tunggal + sigmoid, dilatih dari data Excel. Alur: unggah data → latih → prediksi → laporan.")

with st.sidebar:
    st.subheader("Pengaturan model")
    lr = st.number_input("Learning rate", 0.001, 5.0, 0.3, 0.05, format="%.3f")
    epochs = st.slider("Epoch", 10, 1000, 100, 10)
    ambang = st.slider("Ambang keputusan", 0.1, 0.9, 0.5, 0.05)
    acak = st.checkbox("Acak urutan data tiap epoch", value=False)
    st.caption("Bobot awal: [0.10, 0.20, -0.15], bias awal: 0.05 (sama seperti modul).")

tab1, tab2, tab3, tab4 = st.tabs(["1 · Data", "2 · Latih", "3 · Prediksi", "4 · Laporan"])

# --- Tab 1: data latih
with tab1:
    st.write(f"Format kolom: `{'`, `'.join(FITUR)}`, `{LABEL}` (0 = normal, 1 = cacat). Nilai fitur 0–1.")
    st.download_button("Unduh contoh data (.xlsx)", excel_bytes({"data": data_contoh()}), "data_contoh.xlsx", key="dl_contoh")
    f = st.file_uploader("Unggah data latih (Excel)", type=["xlsx", "xls"], key="up_latih")
    # Proses ulang & reset model hanya jika file latih benar-benar berganti,
    # bukan setiap kali halaman dijalankan ulang (mis. saat unggah file uji).
    if f and ss.get("sig_latih") != (f.name, f.size):
        try:
            df = baca_excel(f)
            if cek_kolom(df, True):
                ss["df_latih"] = df[FITUR + [LABEL]].copy()
                ss["sig_latih"] = (f.name, f.size)
                for k in ("w", "b", "riwayat", "param", "hasil"):
                    ss.pop(k, None)
        except Exception as e:
            st.error(f"File tidak bisa dibaca: {e}")
    if "df_latih" in ss:
        d = ss["df_latih"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah baris", len(d))
        c2.metric("Cacat (1)", int((d[LABEL] == 1).sum()))
        c3.metric("Normal (0)", int((d[LABEL] == 0).sum()))
        st.dataframe(d, use_container_width=True, height=260)

# --- Tab 2: latih
with tab2:
    if "df_latih" not in ss:
        st.info("Unggah data latih di tab 1 terlebih dulu.")
    else:
        if st.button("Mulai training"):
            d = ss["df_latih"]
            with st.spinner("Melatih neuron..."):
                w, b, riw = latih(d[FITUR].values.tolist(), d[LABEL].astype(int).tolist(), [0.10, 0.20, -0.15], 0.05, lr, epochs, acak, 42)
            ss.update(w=w, b=b, riwayat=riw, param=dict(lr=lr, epochs=epochs, ambang=ambang, n_latih=len(d)))
            ss.pop("hasil", None)
        if "w" in ss:
            riw = ss["riwayat"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Loss akhir", f"{riw[-1]['loss']:.4f}", f"{riw[-1]['loss'] - riw[0]['loss']:.4f}", delta_color="inverse")
            c2.metric("Akurasi latih", f"{riw[-1]['akurasi']:.1%}")
            c3.metric("Bias", f"{ss['b']:.3f}")
            rd = pd.DataFrame(riw).set_index("epoch")
            st.line_chart(rd[["loss"]], color=BIRU, height=240)
            st.dataframe(pd.DataFrame({"fitur": FITUR, "bobot": [round(x, 4) for x in ss["w"]]}), hide_index=True, use_container_width=True)
            if ss["param"]["lr"] != lr or ss["param"]["epochs"] != epochs:
                st.caption("Pengaturan di sidebar berubah sejak training terakhir, latih ulang untuk menerapkannya.")

# --- Tab 3: prediksi
with tab3:
    if "w" not in ss:
        st.info("Latih model di tab 2 terlebih dulu.")
    else:
        mode = st.radio("Sumber input", ["Unggah Excel", "Input manual"], horizontal=True)
        if mode == "Input manual":
            c1, c2, c3 = st.columns(3)
            v = [c1.slider("Keretakan", 0.0, 1.0, 0.8, 0.01),
                c2.slider("Ketidakrataan alur", 0.0, 1.0, 0.6, 0.01),
                c3.slider("Beda warna", 0.0, 1.0, 0.5, 0.01)]
            p = skor(v, ss["w"], ss["b"])
            cacat = p >= ambang
            warna = BIRU if cacat else HIJAU
            st.markdown(f"<div class='kartu'><span class='badge' style='background:{warna}'>"
                        f"{'CACAT' if cacat else 'NORMAL'}</span> &nbsp; probabilitas cacat "
                        f"<b>{p:.1%}</b></div>", unsafe_allow_html=True)
            ss["hasil"] = prediksi_df(pd.DataFrame([dict(zip(FITUR, v))]), ss["w"], ss["b"], ambang)
        else:
            st.write(f"Kolom wajib: `{'`, `'.join(FITUR)}`. Kolom `{LABEL}` opsional (untuk menghitung akurasi).")
            f2 = st.file_uploader("Unggah data ban yang akan diprediksi", type=["xlsx", "xls"], key="up_uji")
            if f2:
                try:
                    du = baca_excel(f2)
                    ada_label = LABEL in du.columns
                    if cek_kolom(du, ada_label):
                        ss["hasil"] = prediksi_df(du, ss["w"], ss["b"], ambang)
                except Exception as e:
                    st.error(f"File tidak bisa dibaca: {e}")
            h = ss.get("hasil")
            if f2 and h is not None:
                n_c = int((h["prediksi"] == "CACAT").sum())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(h))
                c2.metric("Cacat", n_c)
                c3.metric("Akurasi", f"{h['benar'].mean():.1%}" if "benar" in h.columns else "-")
                st.dataframe(h, use_container_width=True, height=300)

# --- Tab 4: laporan
with tab4:
    h = ss.get("hasil")
    if "w" not in ss or h is None:
        st.info("Lakukan prediksi di tab 3 untuk membuat laporan.")
    else:
        st.write(f"Laporan mencakup {len(h)} hasil prediksi terakhir, parameter model, dan riwayat loss.")
        p = ss["param"]
        ringkas = pd.DataFrame({
            "item": ["Waktu", "Learning rate", "Epoch", "Ambang", "Data latih", "Bobot akhir", "Bias akhir", "Loss awal", "Loss akhir", "Ban diperiksa", "Terdeteksi cacat"],
            "nilai": [f"{datetime.now():%Y-%m-%d %H:%M}", p["lr"], p["epochs"], p["ambang"], p["n_latih"],
                    str([round(x, 4) for x in ss["w"]]), round(ss["b"], 4),
                    round(ss["riwayat"][0]["loss"], 4), round(ss["riwayat"][-1]["loss"], 4),
                    len(h), int((h["prediksi"] == "CACAT").sum())]})
        xlsx = excel_bytes({"Ringkasan": ringkas, "Hasil Prediksi": h,
                            "Riwayat Loss": pd.DataFrame(ss["riwayat"])})
        c1, c2, c3 = st.columns(3)
        c1.download_button("Laporan Excel", xlsx, "laporan_deteksi_ban.xlsx", key="dl_x")
        c2.download_button("Laporan HTML", laporan_html(ss, h).encode("utf-8"), "laporan_deteksi_ban.html", mime="text/html", key="dl_h")
        c3.download_button("Hasil CSV", h.to_csv(index=False).encode("utf-8"), "hasil_prediksi.csv", mime="text/csv", key="dl_c")