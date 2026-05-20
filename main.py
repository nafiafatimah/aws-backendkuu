from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import pymysql
import json
from datetime import datetime

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI(
    title="EWS Mahasiswa API",
    version="1.0.0"
)

# =====================================================
# CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MODEL
# =====================================================
model_kelulusan = joblib.load("model_rf_kelulusan.pkl")
imputer_kelulusan = joblib.load("imputer.pkl")

model_semester = joblib.load("model_rf_semester.pkl")
imputer_semester = joblib.load("imputer_semester.pkl")

# =====================================================
# FEATURE NAMES SESUAI TRAINING MODEL
# =====================================================

fitur_kelulusan = [
    "ANGKATAN",
    "LABEL_PROG",
    "GENDER_ENC",
    "IPS_MEAN",
    "SKS_MEAN",
    "IPK_MEAN",
    "ABSEN_MEAN",
    "TEORI_MEAN",
    "PRAK_MEAN",
    "CNT_TEPAT_WAKTU"
]

fitur_semester = [
    "SEMESTER_KE",
    "LABEL_PROG",
    "IPS",
    "IPS_DIFF",
    "PROSEN_HADIR_RATA_RATA",
    "TEORI",
    "PRAK",
    "SKS_TARGET",
    "DELAY_TUGAS",
    "TOTAL_SKS_HUTANG",
    "EKONOMI_LEVEL",
    "SKOR_MASUK"
]

# =====================================================
# DATABASE CONNECTION
# =====================================================
def get_db():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="ews_mahasiswa",
        cursorclass=pymysql.cursors.DictCursor
    )

# =====================================================
# REQUEST MODEL
# =====================================================
class KelulusanRequest(BaseModel):
    nrp: str
    nama: str
    angkatan: int
    label_prog: int
    gender_enc: int
    ips_mean: float
    sks_mean: float
    ipk_mean: float
    absen_mean: float
    teori_mean: float
    prak_mean: float
    cnt_tepat_waktu: int


class SemesterRequest(BaseModel):
    nrp: str
    nama: str
    semester_ke: int
    label_prog: int
    ips: float
    ips_diff: float
    presensi: float
    teori: float
    prak: float
    sks_target: int
    delay_tugas: int
    total_sks_hutang: int
    ekonomi_level: int
    skor_masuk: float

# =====================================================
# HOME
# =====================================================
@app.get("/")
def home():
    return {
        "message": "Backend EWS aktif 🚀",
        "timestamp": datetime.now()
    }

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
def get_feature_importance(model, feature_names):

    importance = model.feature_importances_

    hasil = []

    for fitur, nilai in zip(feature_names, importance):

        hasil.append({
            "feature": fitur,
            "importance": round(float(nilai), 5)
        })

    hasil = sorted(
        hasil,
        key=lambda x: x["importance"],
        reverse=True
    )

    return hasil

# =====================================================
# XAI FUNCTION
# =====================================================
def get_xai_explanation(feature_names, input_values, model):

    importances = model.feature_importances_

    hasil = []

    for fitur, nilai_input, bobot in zip(
        feature_names,
        input_values,
        importances
    ):

        kontribusi = float(nilai_input) * float(bobot)

        hasil.append({
            "feature": fitur.lower(),
            "value": float(nilai_input),
            "importance": round(float(bobot), 5),
            "contribution_score": round(kontribusi, 5)
        })

    hasil = sorted(
        hasil,
        key=lambda x: abs(x["contribution_score"]),
        reverse=True
    )

    return hasil
# =====================================================
# =====================================================
# CRUD MAHASISWA KELULUSAN
# =====================================================
# =====================================================

# =====================================================
# CREATE
# =====================================================
@app.post("/mahasiswa/kelulusan")
def create_mahasiswa_kelulusan(data: KelulusanRequest):

    conn = get_db()
    cursor = conn.cursor()

    try:

        cursor.execute("""
        INSERT INTO mahasiswa_aktif
        (
            nrp,
            nama,
            angkatan,
            label_prog,
            gender_enc,
            ips_mean,
            sks_mean,
            ipk_mean,
            absen_mean,
            teori_mean,
            prak_mean,
            cnt_tepat_waktu
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.nrp,
            data.nama,
            data.angkatan,
            data.label_prog,
            data.gender_enc,
            data.ips_mean,
            data.sks_mean,
            data.ipk_mean,
            data.absen_mean,
            data.teori_mean,
            data.prak_mean,
            data.cnt_tepat_waktu
        ))

        conn.commit()

        return {
            "message": "Data mahasiswa berhasil ditambahkan"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        conn.close()

# =====================================================
# READ ALL
# =====================================================
@app.get("/mahasiswa/kelulusan")
def get_all_mahasiswa_kelulusan():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM mahasiswa_aktif
    ORDER BY created_at DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data

# =====================================================
# READ BY NRP
# =====================================================
@app.get("/mahasiswa/kelulusan/{nrp}")
def get_mahasiswa_kelulusan(nrp: str):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM mahasiswa_aktif
    WHERE nrp=%s
    """, (nrp,))

    data = cursor.fetchone()

    conn.close()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Data tidak ditemukan"
        )

    return data

# =====================================================
# UPDATE
# =====================================================
@app.put("/mahasiswa/kelulusan/{nrp}")
def update_mahasiswa_kelulusan(
    nrp: str,
    data: KelulusanRequest
):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE mahasiswa_aktif
    SET
        nama=%s,
        angkatan=%s,
        label_prog=%s,
        gender_enc=%s,
        ips_mean=%s,
        sks_mean=%s,
        ipk_mean=%s,
        absen_mean=%s,
        teori_mean=%s,
        prak_mean=%s,
        cnt_tepat_waktu=%s
    WHERE nrp=%s
    """, (
        data.nama,
        data.angkatan,
        data.label_prog,
        data.gender_enc,
        data.ips_mean,
        data.sks_mean,
        data.ipk_mean,
        data.absen_mean,
        data.teori_mean,
        data.prak_mean,
        data.cnt_tepat_waktu,
        nrp
    ))

    conn.commit()
    conn.close()

    return {
        "message": "Data berhasil diupdate"
    }

# =====================================================
# DELETE
# =====================================================
@app.delete("/mahasiswa/kelulusan/{nrp}")
def delete_mahasiswa_kelulusan(nrp: str):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM mahasiswa_aktif
    WHERE nrp=%s
    """, (nrp,))

    conn.commit()
    conn.close()

    return {
        "message": "Data berhasil dihapus"
    }

# =====================================================
# PREDIKSI KELULUSAN
# =====================================================
@app.post("/predict/kelulusan")
def predict_kelulusan(data: KelulusanRequest):

    conn = get_db()
    cursor = conn.cursor()

    # =====================================================
    # UPSERT DATA MAHASISWA
    # =====================================================
    cursor.execute("""
    SELECT nrp
    FROM mahasiswa_aktif
    WHERE nrp=%s
    """, (data.nrp,))

    existing = cursor.fetchone()

    if existing:

        cursor.execute("""
        UPDATE mahasiswa_aktif
        SET
            nama=%s,
            angkatan=%s,
            label_prog=%s,
            gender_enc=%s,
            ips_mean=%s,
            sks_mean=%s,
            ipk_mean=%s,
            absen_mean=%s,
            teori_mean=%s,
            prak_mean=%s,
            cnt_tepat_waktu=%s
        WHERE nrp=%s
        """, (
            data.nama,
            data.angkatan,
            data.label_prog,
            data.gender_enc,
            data.ips_mean,
            data.sks_mean,
            data.ipk_mean,
            data.absen_mean,
            data.teori_mean,
            data.prak_mean,
            data.cnt_tepat_waktu,
            data.nrp
        ))

    else:

        cursor.execute("""
        INSERT INTO mahasiswa_aktif
        (
            nrp,
            nama,
            angkatan,
            label_prog,
            gender_enc,
            ips_mean,
            sks_mean,
            ipk_mean,
            absen_mean,
            teori_mean,
            prak_mean,
            cnt_tepat_waktu
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.nrp,
            data.nama,
            data.angkatan,
            data.label_prog,
            data.gender_enc,
            data.ips_mean,
            data.sks_mean,
            data.ipk_mean,
            data.absen_mean,
            data.teori_mean,
            data.prak_mean,
            data.cnt_tepat_waktu
        ))

    # =====================================================
    # PREPARE INPUT
    # =====================================================
    input_data = [
        data.angkatan,
        data.label_prog,
        data.gender_enc,
        data.ips_mean,
        data.sks_mean,
        data.ipk_mean,
        data.absen_mean,
        data.teori_mean,
        data.prak_mean,
        data.cnt_tepat_waktu
    ]

    X = pd.DataFrame(
        [input_data],
        columns=fitur_kelulusan
    )

    X = imputer_kelulusan.transform(X)

    # =====================================================
    # PREDIKSI
    # =====================================================
    pred = model_kelulusan.predict(X)[0]

    prob = model_kelulusan.predict_proba(X)[0][1]

    status = (
        "TEPAT WAKTU"
        if pred == 1
        else "TIDAK TEPAT"
    )

    # =====================================================
    # XAI
    # =====================================================
    xai = get_xai_explanation(
        fitur_kelulusan,
        input_data,
        model_kelulusan
    )

    xai_json = json.dumps(xai)

    # =====================================================
    # SIMPAN HASIL
    # =====================================================
    cursor.execute("""
    INSERT INTO hasil_prediksi_kelulusan
    (
        nrp,
        prediksi,
        probabilitas,
        kategori,
        xai_json
    )
    VALUES (%s,%s,%s,%s,%s)
    """, (
        data.nrp,
        int(pred),
        float(prob),
        status,
        xai_json
    ))

    conn.commit()
    conn.close()

    return {
        "nrp": data.nrp,
        "nama": data.nama,
        "status": status,
        "probabilitas": round(float(prob), 4),
        "xai_feature_contribution": xai[:5]
    }

# =====================================================
# FEATURE IMPORTANCE KELULUSAN
# =====================================================
@app.get("/xai/feature-importance/kelulusan")
def feature_importance_kelulusan():

    hasil = get_feature_importance(
        model_kelulusan,
        fitur_kelulusan
    )

    return {
        "model": "Random Forest Kelulusan",
        "feature_importance": hasil
    }

# =====================================================
# RESULT KELULUSAN
# =====================================================
@app.get("/results/kelulusan")
def get_kelulusan():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        m.nrp,
        m.nama,
        h.prediksi,
        h.kategori,
        h.probabilitas,
        h.xai_json,
        h.created_at
    FROM hasil_prediksi_kelulusan h
    JOIN mahasiswa_aktif m
    ON m.nrp = h.nrp
    ORDER BY h.created_at DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data

# =====================================================
# RESULT KELULUSAN BY NRP
# =====================================================
@app.get("/results/kelulusan/{nrp}")
def get_kelulusan_by_nrp(nrp: str):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        m.nrp,
        m.nama,
        h.prediksi,
        h.kategori,
        h.probabilitas,
        h.xai_json,
        h.created_at
    FROM hasil_prediksi_kelulusan h
    JOIN mahasiswa_aktif m
    ON m.nrp = h.nrp
    WHERE m.nrp=%s
    ORDER BY h.created_at DESC
    LIMIT 1
    """, (nrp,))

    data = cursor.fetchone()

    conn.close()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Data tidak ditemukan"
        )

    return data

# =====================================================
# =====================================================
# CRUD SEMESTER
# =====================================================
# =====================================================

# =====================================================
# CREATE
# =====================================================
@app.post("/mahasiswa/semester")
def create_semester(data: SemesterRequest):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO mahasiswa_aktif_semester
    (
        nrp,
        nama,
        semester_ke,
        label_prog,
        ips,
        ips_diff,
        presensi,
        teori,
        prak,
        sks_target,
        delay_tugas,
        total_sks_hutang,
        ekonomi_level,
        skor_masuk
    )
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        data.nrp,
        data.nama,
        data.semester_ke,
        data.label_prog,
        data.ips,
        data.ips_diff,
        data.presensi,
        data.teori,
        data.prak,
        data.sks_target,
        data.delay_tugas,
        data.total_sks_hutang,
        data.ekonomi_level,
        data.skor_masuk
    ))

    conn.commit()
    conn.close()

    return {
        "message": "Data semester berhasil ditambahkan"
    }

# =====================================================
# READ ALL
# =====================================================
@app.get("/mahasiswa/semester")
def get_all_semester():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM mahasiswa_aktif_semester
    ORDER BY created_at DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data

# =====================================================
# READ BY NRP
# =====================================================
@app.get("/mahasiswa/semester/{nrp}")
def get_semester_by_nrp(nrp: str):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT *
    FROM mahasiswa_aktif_semester
    WHERE nrp=%s
    """, (nrp,))

    data = cursor.fetchone()

    conn.close()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Data tidak ditemukan"
        )

    return data

# =====================================================
# UPDATE
# =====================================================
@app.put("/mahasiswa/semester/{nrp}")
def update_semester(
    nrp: str,
    data: SemesterRequest
):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE mahasiswa_aktif_semester
    SET
        nama=%s,
        semester_ke=%s,
        label_prog=%s,
        ips=%s,
        ips_diff=%s,
        presensi=%s,
        teori=%s,
        prak=%s,
        sks_target=%s,
        delay_tugas=%s,
        total_sks_hutang=%s,
        ekonomi_level=%s,
        skor_masuk=%s
    WHERE nrp=%s
    """, (
        data.nama,
        data.semester_ke,
        data.label_prog,
        data.ips,
        data.ips_diff,
        data.presensi,
        data.teori,
        data.prak,
        data.sks_target,
        data.delay_tugas,
        data.total_sks_hutang,
        data.ekonomi_level,
        data.skor_masuk,
        nrp
    ))

    conn.commit()
    conn.close()

    return {
        "message": "Data semester berhasil diupdate"
    }

# =====================================================
# DELETE
# =====================================================
@app.delete("/mahasiswa/semester/{nrp}")
def delete_semester(nrp: str):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    DELETE FROM mahasiswa_aktif_semester
    WHERE nrp=%s
    """, (nrp,))

    conn.commit()
    conn.close()

    return {
        "message": "Data semester berhasil dihapus"
    }

# =====================================================
# PREDIKSI SEMESTER
# =====================================================
@app.post("/predict/semester")
def predict_semester(data: SemesterRequest):

    conn = get_db()
    cursor = conn.cursor()

    # =====================================================
    # UPSERT DATA MAHASISWA
    # =====================================================
    cursor.execute("""
    SELECT nrp
    FROM mahasiswa_aktif_semester
    WHERE nrp=%s
    """, (data.nrp,))

    existing = cursor.fetchone()

    if existing:

        cursor.execute("""
        UPDATE mahasiswa_aktif_semester
        SET
            nama=%s,
            semester_ke=%s,
            label_prog=%s,
            ips=%s,
            ips_diff=%s,
            presensi=%s,
            teori=%s,
            prak=%s,
            sks_target=%s,
            delay_tugas=%s,
            total_sks_hutang=%s,
            ekonomi_level=%s,
            skor_masuk=%s
        WHERE nrp=%s
        """, (
            data.nama,
            data.semester_ke,
            data.label_prog,
            data.ips,
            data.ips_diff,
            data.presensi,
            data.teori,
            data.prak,
            data.sks_target,
            data.delay_tugas,
            data.total_sks_hutang,
            data.ekonomi_level,
            data.skor_masuk,
            data.nrp
        ))

    else:

        cursor.execute("""
        INSERT INTO mahasiswa_aktif_semester
        (
            nrp,
            nama,
            semester_ke,
            label_prog,
            ips,
            ips_diff,
            presensi,
            teori,
            prak,
            sks_target,
            delay_tugas,
            total_sks_hutang,
            ekonomi_level,
            skor_masuk
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.nrp,
            data.nama,
            data.semester_ke,
            data.label_prog,
            data.ips,
            data.ips_diff,
            data.presensi,
            data.teori,
            data.prak,
            data.sks_target,
            data.delay_tugas,
            data.total_sks_hutang,
            data.ekonomi_level,
            data.skor_masuk
        ))

    # =====================================================
    # PREPARE INPUT
    # =====================================================
    input_data = [
        data.semester_ke,
        data.label_prog,
        data.ips,
        data.ips_diff,
        data.presensi,
        data.teori,
        data.prak,
        data.sks_target,
        data.delay_tugas,
        data.total_sks_hutang,
        data.ekonomi_level,
        data.skor_masuk
    ]

    X = pd.DataFrame(
        [input_data],
        columns=fitur_semester
    )

    X = imputer_semester.transform(X)

    # =====================================================
    # PREDIKSI
    # =====================================================
    pred = model_semester.predict(X)[0]

    prob = model_semester.predict_proba(X)[0][1]

    status = (
        "AMAN"
        if pred == 1
        else "RISIKO TINGGI"
    )

    # =====================================================
    # XAI
    # =====================================================
    xai = get_xai_explanation(
        fitur_semester,
        input_data,
        model_semester
    )

    xai_json = json.dumps(xai)

    # =====================================================
    # SIMPAN HASIL
    # =====================================================
    cursor.execute("""
    INSERT INTO hasil_prediksi_semester
    (
        nrp,
        prediksi,
        probabilitas,
        kategori,
        xai_json
    )
    VALUES (%s,%s,%s,%s,%s)
    """, (
        data.nrp,
        int(pred),
        float(prob),
        status,
        xai_json
    ))

    conn.commit()
    conn.close()

    return {
        "nrp": data.nrp,
        "nama": data.nama,
        "status": status,
        "probabilitas": round(float(prob), 4),
        "xai_feature_contribution": xai[:5]
    }

# =====================================================
# FEATURE IMPORTANCE SEMESTER
# =====================================================
@app.get("/xai/feature-importance/semester")
def feature_importance_semester():

    hasil = get_feature_importance(
        model_semester,
        fitur_semester
    )

    return {
        "model": "Random Forest Semester",
        "feature_importance": hasil
    }

# =====================================================
# RESULT SEMESTER
# =====================================================
@app.get("/results/semester")
def get_semester():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        m.nrp,
        m.nama,
        h.prediksi,
        h.kategori,
        h.probabilitas,
        h.xai_json,
        h.created_at
    FROM hasil_prediksi_semester h
    JOIN mahasiswa_aktif_semester m
    ON m.nrp = h.nrp
    ORDER BY h.created_at DESC
    """)

    data = cursor.fetchall()

    conn.close()

    return data

# =====================================================
# RESULT SEMESTER BY NRP
# =====================================================
@app.get("/results/semester/{nrp}")
def get_result_semester_by_nrp(nrp: str):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        m.nrp,
        m.nama,
        h.prediksi,
        h.kategori,
        h.probabilitas,
        h.xai_json,
        h.created_at
    FROM hasil_prediksi_semester h
    JOIN mahasiswa_aktif_semester m
    ON m.nrp = h.nrp
    WHERE m.nrp=%s
    ORDER BY h.created_at DESC
    LIMIT 1
    """, (nrp,))

    data = cursor.fetchone()

    conn.close()

    if not data:

        raise HTTPException(
            status_code=404,
            detail="Data tidak ditemukan"
        )

    return data

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health_check():

    return {
        "status": "OK",
        "server_time": datetime.now()
    }