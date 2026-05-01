from fastapi import FastAPI
import joblib
import numpy as np
import pymysql

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # bebas dulu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# LOAD MODEL
# =========================
model_kelulusan = joblib.load("model_rf_kelulusan.pkl")
imputer_kelulusan = joblib.load("imputer.pkl")

model_semester = joblib.load("model_rf_semester.pkl")
imputer_semester = joblib.load("imputer_semester.pkl")

# =========================
# KONEKSI DATABASE
# =========================
def get_db():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",  # sesuaikan
        database="ews_mahasiswa",
        cursorclass=pymysql.cursors.DictCursor
    )

# =========================
# CEK SERVER
# =========================
@app.get("/")
def home():
    return {"message": "Backend EWS aktif 🚀"}

# =====================================================
# 🎓 ENDPOINT 1: PREDIKSI KELULUSAN
# =====================================================
@app.post("/predict/kelulusan")
def predict_kelulusan(data: dict):

    conn = get_db()
    cursor = conn.cursor()

    # =========================
    # 1. SIMPAN INPUT
    # =========================
    cursor.execute("""
    INSERT INTO mahasiswa_aktif
    (nrp, nama, angkatan, label_prog, gender_enc,
     ips_mean, sks_mean, ipk_mean, absen_mean,
     teori_mean, prak_mean, cnt_tepat_waktu)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        data["nrp"],
        data["nama"],
        data["angkatan"],
        data["label_prog"],
        data["gender_enc"],
        data["ips_mean"],
        data["sks_mean"],
        data["ipk_mean"],
        data["absen_mean"],
        data["teori_mean"],
        data["prak_mean"],
        data["cnt_tepat_waktu"]
    ))

    # =========================
    # 2. PREPARE MODEL INPUT
    # =========================
    X = np.array([[ 
        data["angkatan"],
        data["label_prog"],
        data["gender_enc"],
        data["ips_mean"],
        data["sks_mean"],
        data["ipk_mean"],
        data["absen_mean"],
        data["teori_mean"],
        data["prak_mean"],
        data["cnt_tepat_waktu"]
    ]])

    X = imputer_kelulusan.transform(X)

    # =========================
    # 3. PREDIKSI
    # =========================
    pred = model_kelulusan.predict(X)[0]
    prob = model_kelulusan.predict_proba(X)[0][1]

    status = "TEPAT WAKTU" if pred == 1 else "TIDAK TEPAT"

    # =========================
    # 4. SIMPAN HASIL
    # =========================
    cursor.execute("""
    INSERT INTO hasil_prediksi_kelulusan
    (nrp, prediksi, probabilitas, kategori)
    VALUES (%s,%s,%s,%s)
    """, (
        data["nrp"],
        int(pred),
        float(prob),
        status
    ))

    conn.commit()
    conn.close()

    return {
        "nrp": data["nrp"],
        "nama": data["nama"],
        "status": status,
        "probabilitas": float(prob)
    }

@app.get("/results/kelulusan/{nrp}")
def get_kelulusan_by_nrp(nrp: str):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT m.nrp, m.nama, h.kategori, h.probabilitas, h.created_at
    FROM hasil_prediksi_kelulusan h
    JOIN mahasiswa_aktif m ON m.nrp = h.nrp
    WHERE m.nrp = %s
    ORDER BY h.created_at DESC
    LIMIT 1
    """, (nrp,))

    data = cursor.fetchone()
    conn.close()

    if not data:
        return {"message": "Data tidak ditemukan"}

    return data
    
# =====================================================
# 📊 ENDPOINT 2: PREDIKSI SEMESTER (EWS)
# =====================================================
@app.post("/predict/semester")
def predict_semester(data: dict):

    conn = get_db()
    cursor = conn.cursor()

    # =========================
    # 1. SIMPAN INPUT
    # =========================
    cursor.execute("""
    INSERT INTO mahasiswa_aktif_semester
    (nrp, nama, semester_ke, label_prog, ips, ips_diff,
     presensi, teori, prak, sks_target,
     delay_tugas, total_sks_hutang,
     ekonomi_level, skor_masuk)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        data["nrp"],
        data["nama"],
        data["semester_ke"],
        data["label_prog"],
        data["ips"],
        data["ips_diff"],
        data["presensi"],
        data["teori"],
        data["prak"],
        data["sks_target"],
        data["delay_tugas"],
        data["total_sks_hutang"],
        data["ekonomi_level"],
        data["skor_masuk"]
    ))

    # =========================
    # 2. PREPARE INPUT MODEL
    # =========================
    X = np.array([[ 
        data["semester_ke"],
        data["label_prog"],
        data["ips"],
        data["ips_diff"],
        data["presensi"],
        data["teori"],
        data["prak"],
        data["sks_target"],
        data["delay_tugas"],
        data["total_sks_hutang"],
        data["ekonomi_level"],
        data["skor_masuk"]
    ]])

    X = imputer_semester.transform(X)

    # =========================
    # 3. PREDIKSI
    # =========================
    pred = model_semester.predict(X)[0]
    prob = model_semester.predict_proba(X)[0][1]

    status = "AMAN" if pred == 1 else "RISIKO TINGGI"

    # =========================
    # 4. SIMPAN HASIL
    # =========================
    cursor.execute("""
    INSERT INTO hasil_prediksi_semester
    (nrp, prediksi, probabilitas, kategori)
    VALUES (%s,%s,%s,%s)
    """, (
        data["nrp"],
        int(pred),
        float(prob),
        status
    ))

    conn.commit()
    conn.close()

    return {
        "nrp": data["nrp"],
        "nama": data["nama"],
        "status": status,
        "probabilitas": float(prob)
    }


# =====================================================
# 📂 GET DATA HASIL
# =====================================================
@app.get("/results/kelulusan")
def get_kelulusan():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT m.nrp, m.nama, h.kategori, h.probabilitas, h.created_at
    FROM hasil_prediksi_kelulusan h
    JOIN mahasiswa_aktif m ON m.nrp = h.nrp
    ORDER BY h.created_at DESC
    """)

    data = cursor.fetchall()
    conn.close()

    return data

@app.get("/results/kelulusan/{nrp}")
def get_kelulusan_by_nrp(nrp: str):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT m.nrp, m.nama, h.kategori, h.probabilitas, h.created_at
    FROM hasil_prediksi_kelulusan h
    JOIN mahasiswa_aktif m ON m.nrp = h.nrp
    WHERE m.nrp = %s
    ORDER BY h.created_at DESC
    LIMIT 1
    """, (nrp,))

    data = cursor.fetchone()
    conn.close()

    if not data:
        return {"message": "Data tidak ditemukan"}

    return data


@app.get("/results/semester")
def get_semester():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT m.nrp, m.nama, h.kategori, h.probabilitas, h.created_at
    FROM hasil_prediksi_semester h
    JOIN mahasiswa_aktif_semester m ON m.nrp = h.nrp
    ORDER BY h.created_at DESC
    """)

    data = cursor.fetchall()
    conn.close()

    return data

@app.get("/results/semester/{nrp}")
def get_semester_by_nrp(nrp: str):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT m.nrp, m.nama, h.kategori, h.probabilitas, h.created_at
    FROM hasil_prediksi_semester h
    JOIN mahasiswa_aktif_semester m ON m.nrp = h.nrp
    WHERE m.nrp = %s
    ORDER BY h.created_at DESC
    LIMIT 1
    """, (nrp,))

    data = cursor.fetchone()
    conn.close()

    if not data:
        return {"message": "Data tidak ditemukan"}

    return data

