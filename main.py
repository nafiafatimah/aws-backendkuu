import io
import joblib
import json
from datetime import datetime
import numpy as np
import pandas as pd
import pymysql
import bcrypt
from jose import jwt
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

# =====================================================
# FASTAPI CONFIGURATION
# =====================================================
app = FastAPI(
    title="EWS Mahasiswa API",
    version="1.0.0"
)

# Konfigurasi Keamanan JWT (Silakan ganti secret key sesuai kebutuhan)
SECRET_KEY = "SUPER_SECRET_EWS_KEY_A_XYZ"
ALGORITHM = "HS256"

# =====================================================
# CORS MIDDLEWARE
# =====================================================
ORIGINS = [
    "http://localhost:3000",      
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MACHINE LEARNING MODELS
# =====================================================
model_kelulusan = joblib.load("model_rf_kelulusan.pkl")
imputer_kelulusan = joblib.load("imputer.pkl")

model_semester = joblib.load("model_rf_semester.pkl")
imputer_semester = joblib.load("imputer_semester.pkl")

# =====================================================
# FEATURE NAMES (MUST MATCH MODEL TRAINING)
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
# DATABASE CONNECTION FACTORY
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
# PYDANTIC REQUEST MODELS
# =====================================================
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

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
# UPDATE REQUEST MODELS
# =====================================================
class UpdateKelulusanRequest(BaseModel):
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

class UpdateSemesterRequest(BaseModel):
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
# UTILITY HELPER FUNCTIONS (CORE & AUTH)
# =====================================================
def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_feature_importance(model, feature_names):
    importance = model.feature_importances_
    hasil = []
    for fitur, nilai in zip(feature_names, importance):
        hasil.append({
            "feature": fitur,
            "importance": round(float(nilai), 5)
        })
    return sorted(hasil, key=lambda x: x["importance"], reverse=True)

def get_xai_explanation(feature_names, input_values, model):
    importances = model.feature_importances_
    hasil = []
    for fitur, nilai_input, bobot in zip(feature_names, input_values, importances):
        kontribusi = float(nilai_input) * float(bobot)
        hasil.append({
            "feature": fitur.lower(),
            "value": float(nilai_input),
            "importance": round(float(bobot), 5),
            "contribution_score": round(kontribusi, 5)
        })
    return sorted(hasil, key=lambda x: abs(x["contribution_score"]), reverse=True)

# =====================================================
# SYSTEM MAIN ENDPOINTS / HOME
# =====================================================
@app.get("/")
def home():
    return {
        "message": "Backend EWS aktif 🚀",
        "timestamp": datetime.now()
    }

# =====================================================
# AUTHENTICATION ENDPOINTS
# =====================================================
@app.post("/auth/register")
def register_user(data: RegisterRequest):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Pengecekan apakah kredensial duplikat
        cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (data.username, data.email))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username atau Email sudah terdaftar")
        
        hashed_pwd = hash_password(data.password)
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (data.username, data.email, hashed_pwd)
        )
        conn.commit()
        return {"status": "success", "message": "Registrasi berhasil! Silakan login."}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/auth/login")
def login_user(data: LoginRequest):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (data.username,))
        user = cursor.fetchone()
        
        if not user or not verify_password(data.password, user["password"]):
            raise HTTPException(status_code=401, detail="Username atau password salah")
        
        token = create_access_token({"sub": user["username"], "role": user["role"]})
        return {
            "status": "success",
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "username": user["username"],
                "email": user["email"],
                "role": user["role"]
            }
        }
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# =====================================================
# CRUD MAHASISWA KELULUSAN
# =====================================================

# CREATE - Kelulusan
@app.post("/mahasiswa/kelulusan")
def create_mahasiswa_kelulusan(data: KelulusanRequest):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Cek apakah NRP sudah ada
        cursor.execute("SELECT nrp FROM mahasiswa_aktif WHERE nrp = %s", (data.nrp,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail=f"NRP {data.nrp} sudah terdaftar")
        
        cursor.execute("""
        INSERT INTO mahasiswa_aktif 
        (nrp, nama, angkatan, label_prog, gender_enc, ips_mean, sks_mean, ipk_mean, absen_mean, teori_mean, prak_mean, cnt_tepat_waktu)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.nrp, data.nama, data.angkatan, data.label_prog, data.gender_enc,
            data.ips_mean, data.sks_mean, data.ipk_mean, data.absen_mean,
            data.teori_mean, data.prak_mean, data.cnt_tepat_waktu
        ))
        conn.commit()
        return {"status": "success", "message": "Data mahasiswa berhasil ditambahkan"}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# READ ALL - Kelulusan
@app.get("/mahasiswa/kelulusan")
def get_all_mahasiswa_kelulusan():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mahasiswa_aktif ORDER BY created_at DESC")
    data = cursor.fetchall()
    conn.close()
    return data

# READ ONE - Kelulusan
@app.get("/mahasiswa/kelulusan/{nrp}")
def get_mahasiswa_kelulusan(nrp: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mahasiswa_aktif WHERE nrp=%s", (nrp,))
    data = cursor.fetchone()
    conn.close()
    if not data:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    return data

# UPDATE - Kelulusan
@app.put("/mahasiswa/kelulusan/{nrp}")
def update_mahasiswa_kelulusan(nrp: str, data: UpdateKelulusanRequest):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Cek apakah data ada
        cursor.execute("SELECT nrp FROM mahasiswa_aktif WHERE nrp = %s", (nrp,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Mahasiswa dengan NRP {nrp} tidak ditemukan")

        # Update data
        cursor.execute("""
        UPDATE mahasiswa_aktif SET
            nama=%s, angkatan=%s, label_prog=%s, gender_enc=%s,
            ips_mean=%s, sks_mean=%s, ipk_mean=%s, absen_mean=%s,
            teori_mean=%s, prak_mean=%s, cnt_tepat_waktu=%s
        WHERE nrp=%s
        """, (
            data.nama, data.angkatan, data.label_prog, data.gender_enc,
            data.ips_mean, data.sks_mean, data.ipk_mean, data.absen_mean,
            data.teori_mean, data.prak_mean, data.cnt_tepat_waktu, nrp
        ))
        conn.commit()
        
        return {
            "status": "success",
            "message": f"Data mahasiswa dengan NRP {nrp} berhasil diperbarui"
        }
    except pymysql.MySQLError as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
    finally:
        conn.close()

# DELETE - Kelulusan
@app.delete("/mahasiswa/kelulusan/{nrp}")
def delete_mahasiswa_kelulusan(nrp: str):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Cek apakah data ada
        cursor.execute("SELECT nrp FROM mahasiswa_aktif WHERE nrp = %s", (nrp,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Mahasiswa dengan NRP {nrp} tidak ditemukan")

        # Hapus data
        cursor.execute("DELETE FROM mahasiswa_aktif WHERE nrp = %s", (nrp,))
        conn.commit()
        
        return {
            "status": "success",
            "message": f"Data mahasiswa dengan NRP {nrp} berhasil dihapus"
        }
    except pymysql.MySQLError as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
    finally:
        conn.close()

# =====================================================
# PREDIKSI & ADVANCED XAI KELULUSAN
# =====================================================
@app.post("/predict/kelulusan")
def predict_kelulusan(data: KelulusanRequest):
    conn = get_db()
    cursor = conn.cursor()
    
    # Perilaku Upsert Kompatibel
    cursor.execute("SELECT nrp FROM mahasiswa_aktif WHERE nrp=%s", (data.nrp,))
    if cursor.fetchone():
        cursor.execute("""
        UPDATE mahasiswa_aktif SET
            nama=%s, angkatan=%s, label_prog=%s, gender_enc=%s, ips_mean=%s,
            sks_mean=%s, ipk_mean=%s, absen_mean=%s, teori_mean=%s, prak_mean=%s, cnt_tepat_waktu=%s
        WHERE nrp=%s
        """, (
            data.nama, data.angkatan, data.label_prog, data.gender_enc, data.ips_mean,
            data.sks_mean, data.ipk_mean, data.absen_mean, data.teori_mean, data.prak_mean,
            data.cnt_tepat_waktu, data.nrp
        ))
    else:
        cursor.execute("""
        INSERT INTO mahasiswa_aktif (nrp, nama, angkatan, label_prog, gender_enc, ips_mean, sks_mean, ipk_mean, absen_mean, teori_mean, prak_mean, cnt_tepat_waktu)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.nrp, data.nama, data.angkatan, data.label_prog, data.gender_enc, data.ips_mean,
            data.sks_mean, data.ipk_mean, data.absen_mean, data.teori_mean, data.prak_mean, data.cnt_tepat_waktu
        ))

    input_data = [
        data.angkatan, data.label_prog, data.gender_enc, data.ips_mean, data.sks_mean,
        data.ipk_mean, data.absen_mean, data.teori_mean, data.prak_mean, data.cnt_tepat_waktu
    ]
    X = pd.DataFrame([input_data], columns=fitur_kelulusan)
    X = imputer_kelulusan.transform(X)

    pred = model_kelulusan.predict(X)[0]
    prob = model_kelulusan.predict_proba(X)[0][1]
    status = "TEPAT WAKTU" if pred == 1 else "TIDAK TEPAT"

    xai = get_xai_explanation(fitur_kelulusan, input_data, model_kelulusan)
    xai_json = json.dumps(xai)

    cursor.execute("""
    INSERT INTO hasil_prediksi_kelulusan (nrp, prediksi, probabilitas, kategori, xai_json)
    VALUES (%s,%s,%s,%s,%s)
    """, (data.nrp, int(pred), float(prob), status, xai_json))
    
    conn.commit()
    conn.close()

    return {
        "nrp": data.nrp,
        "nama": data.nama,
        "status": status,
        "probabilitas": round(float(prob), 4),
        "xai_feature_contribution": xai[:5]
    }

@app.get("/xai/feature-importance/kelulusan")
def feature_importance_kelulusan():
    hasil = get_feature_importance(model_kelulusan, fitur_kelulusan)
    return {"model": "Random Forest Kelulusan", "feature_importance": hasil}

@app.get("/results/kelulusan")
def get_kelulusan():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT m.nrp, m.nama, h.prediksi, h.kategori, h.probabilitas, h.xai_json, h.created_at
    FROM hasil_prediksi_kelulusan h JOIN mahasiswa_aktif m ON m.nrp = h.nrp
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
    SELECT m.nrp, m.nama, h.prediksi, h.kategori, h.probabilitas, h.xai_json, h.created_at
    FROM hasil_prediksi_kelulusan h JOIN mahasiswa_aktif m ON m.nrp = h.nrp
    WHERE m.nrp=%s ORDER BY h.created_at DESC LIMIT 1
    """, (nrp,))
    data = cursor.fetchone()
    conn.close()
    if not data:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    return data

# =====================================================
# CRUD MAHASISWA SEMESTER
# =====================================================

# CREATE - Semester
@app.post("/mahasiswa/semester")
def create_semester(data: SemesterRequest):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Cek apakah NRP sudah ada
        cursor.execute("SELECT nrp FROM mahasiswa_aktif_semester WHERE nrp = %s", (data.nrp,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail=f"NRP {data.nrp} sudah terdaftar di data semester")
        
        cursor.execute("""
        INSERT INTO mahasiswa_aktif_semester (nrp, nama, semester_ke, label_prog, ips, ips_diff, presensi, teori, prak, sks_target, delay_tugas, total_sks_hutang, ekonomi_level, skor_masuk)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.nrp, data.nama, data.semester_ke, data.label_prog, data.ips, data.ips_diff,
            data.presensi, data.teori, data.prak, data.sks_target, data.delay_tugas,
            data.total_sks_hutang, data.ekonomi_level, data.skor_masuk
        ))
        conn.commit()
        return {"status": "success", "message": "Data semester berhasil ditambahkan"}
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# READ ALL - Semester
@app.get("/mahasiswa/semester")
def get_all_semester():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mahasiswa_aktif_semester ORDER BY created_at DESC")
    data = cursor.fetchall()
    conn.close()
    return data

# READ ONE - Semester
@app.get("/mahasiswa/semester/{nrp}")
def get_semester_by_nrp(nrp: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM mahasiswa_aktif_semester WHERE nrp=%s", (nrp,))
    data = cursor.fetchone()
    conn.close()
    if not data:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    return data

# UPDATE - Semester
@app.put("/mahasiswa/semester/{nrp}")
def update_mahasiswa_semester(nrp: str, data: UpdateSemesterRequest):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Cek apakah data ada
        cursor.execute("SELECT nrp FROM mahasiswa_aktif_semester WHERE nrp = %s", (nrp,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Data semester mahasiswa dengan NRP {nrp} tidak ditemukan")

        # Update data
        cursor.execute("""
        UPDATE mahasiswa_aktif_semester SET
            nama=%s, semester_ke=%s, label_prog=%s,
            ips=%s, ips_diff=%s, presensi=%s,
            teori=%s, prak=%s, sks_target=%s,
            delay_tugas=%s, total_sks_hutang=%s,
            ekonomi_level=%s, skor_masuk=%s
        WHERE nrp=%s
        """, (
            data.nama, data.semester_ke, data.label_prog,
            data.ips, data.ips_diff, data.presensi,
            data.teori, data.prak, data.sks_target,
            data.delay_tugas, data.total_sks_hutang,
            data.ekonomi_level, data.skor_masuk, nrp
        ))
        conn.commit()
        
        return {
            "status": "success",
            "message": f"Data semester mahasiswa dengan NRP {nrp} berhasil diperbarui"
        }
    except pymysql.MySQLError as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
    finally:
        conn.close()

# DELETE - Semester
@app.delete("/mahasiswa/semester/{nrp}")
def delete_mahasiswa_semester(nrp: str):
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Cek apakah data ada
        cursor.execute("SELECT nrp FROM mahasiswa_aktif_semester WHERE nrp = %s", (nrp,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Data semester mahasiswa dengan NRP {nrp} tidak ditemukan")

        # Hapus data
        cursor.execute("DELETE FROM mahasiswa_aktif_semester WHERE nrp = %s", (nrp,))
        conn.commit()
        
        return {
            "status": "success",
            "message": f"Data semester mahasiswa dengan NRP {nrp} berhasil dihapus"
        }
    except pymysql.MySQLError as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
    finally:
        conn.close()

# =====================================================
# PREDIKSI & ADVANCED XAI SEMESTER
# =====================================================
@app.post("/predict/semester")
def predict_semester(data: SemesterRequest):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT nrp FROM mahasiswa_aktif_semester WHERE nrp=%s", (data.nrp,))
    if cursor.fetchone():
        cursor.execute("""
        UPDATE mahasiswa_aktif_semester SET
            nama=%s, semester_ke=%s, label_prog=%s, ips=%s, ips_diff=%s, presensi=%s,
            teori=%s, prak=%s, sks_target=%s, delay_tugas=%s, total_sks_hutang=%s, ekonomi_level=%s, skor_masuk=%s
        WHERE nrp=%s
        """, (
            data.nama, data.semester_ke, data.label_prog, data.ips, data.ips_diff, data.presensi,
            data.teori, data.prak, data.sks_target, data.delay_tugas, data.total_sks_hutang,
            data.ekonomi_level, data.skor_masuk, data.nrp
        ))
    else:
        cursor.execute("""
        INSERT INTO mahasiswa_aktif_semester (nrp, nama, semester_ke, label_prog, ips, ips_diff, presensi, teori, prak, sks_target, delay_tugas, total_sks_hutang, ekonomi_level, skor_masuk)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data.nrp, data.nama, data.semester_ke, data.label_prog, data.ips, data.ips_diff,
            data.presensi, data.teori, data.prak, data.sks_target, data.delay_tugas,
            data.total_sks_hutang, data.ekonomi_level, data.skor_masuk
        ))

    input_data = [
        data.semester_ke, data.label_prog, data.ips, data.ips_diff, data.presensi,
        data.teori, data.prak, data.sks_target, data.delay_tugas, data.total_sks_hutang,
        data.ekonomi_level, data.skor_masuk
    ]
    X = pd.DataFrame([input_data], columns=fitur_semester)
    X = imputer_semester.transform(X)

    pred = model_semester.predict(X)[0]
    prob = model_semester.predict_proba(X)[0][1]
    status = "AMAN" if pred == 1 else "RISIKO TINGGI"

    xai = get_xai_explanation(fitur_semester, input_data, model_semester)
    xai_json = json.dumps(xai)

    cursor.execute("""
    INSERT INTO hasil_prediksi_semester (nrp, prediksi, probabilitas, kategori, xai_json)
    VALUES (%s,%s,%s,%s,%s)
    """, (data.nrp, int(pred), float(prob), status, xai_json))
    
    conn.commit()
    conn.close()

    return {
        "nrp": data.nrp,
        "nama": data.nama,
        "status": status,
        "probabilitas": round(float(prob), 4),
        "xai_feature_contribution": xai[:5]
    }

@app.get("/xai/feature-importance/semester")
def feature_importance_semester():
    hasil = get_feature_importance(model_semester, fitur_semester)
    return {"model": "Random Forest Semester", "feature_importance": hasil}

@app.get("/results/semester")
def get_semester():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT m.nrp, m.nama, h.prediksi, h.kategori, h.probabilitas, h.xai_json, h.created_at
    FROM hasil_prediksi_semester h JOIN mahasiswa_aktif_semester m ON m.nrp = h.nrp
    ORDER BY h.created_at DESC
    """)
    data = cursor.fetchall()
    conn.close()
    return data

@app.get("/results/semester/{nrp}")
def get_result_semester_by_nrp(nrp: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT m.nrp, m.nama, h.prediksi, h.kategori, h.probabilitas, h.xai_json, h.created_at
    FROM hasil_prediksi_semester h JOIN mahasiswa_aktif_semester m ON m.nrp = h.nrp
    WHERE m.nrp=%s ORDER BY h.created_at DESC LIMIT 1
    """, (nrp,))
    data = cursor.fetchone()
    conn.close()
    if not data:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    return data

# =====================================================
# CSV PREDICTION ENDPOINTS
# =====================================================

# CSV PREDICTION - KELULUSAN
@app.post("/predict/kelulusan/csv")
async def predict_kelulusan_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="File harus CSV"
        )

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_cols = [
        "nrp","nama","angkatan","label_prog","gender_enc",
        "ips_mean","sks_mean","ipk_mean","absen_mean",
        "teori_mean","prak_mean","cnt_tepat_waktu"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom tidak lengkap: {missing}"
        )

    conn = get_db()
    cursor = conn.cursor()

    hasil = []

    try:
        for _, row in df.iterrows():
            nrp = str(row["nrp"])

            cursor.execute(
                "SELECT nrp FROM mahasiswa_aktif WHERE nrp=%s",
                (nrp,)
            )

            if cursor.fetchone():
                cursor.execute("""
                UPDATE mahasiswa_aktif SET
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
                """,(
                    row["nama"],
                    row["angkatan"],
                    row["label_prog"],
                    row["gender_enc"],
                    row["ips_mean"],
                    row["sks_mean"],
                    row["ipk_mean"],
                    row["absen_mean"],
                    row["teori_mean"],
                    row["prak_mean"],
                    row["cnt_tepat_waktu"],
                    nrp
                ))
            else:
                cursor.execute("""
                INSERT INTO mahasiswa_aktif
                (
                    nrp,nama,angkatan,label_prog,
                    gender_enc,ips_mean,sks_mean,
                    ipk_mean,absen_mean,teori_mean,
                    prak_mean,cnt_tepat_waktu
                )
                VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,(
                    nrp,
                    row["nama"],
                    row["angkatan"],
                    row["label_prog"],
                    row["gender_enc"],
                    row["ips_mean"],
                    row["sks_mean"],
                    row["ipk_mean"],
                    row["absen_mean"],
                    row["teori_mean"],
                    row["prak_mean"],
                    row["cnt_tepat_waktu"]
                ))

            input_data = [[
                row["angkatan"],
                row["label_prog"],
                row["gender_enc"],
                row["ips_mean"],
                row["sks_mean"],
                row["ipk_mean"],
                row["absen_mean"],
                row["teori_mean"],
                row["prak_mean"],
                row["cnt_tepat_waktu"]
            ]]

            X = pd.DataFrame(
                input_data,
                columns=fitur_kelulusan
            )

            X = imputer_kelulusan.transform(X)

            pred = model_kelulusan.predict(X)[0]
            prob = model_kelulusan.predict_proba(X)[0][1]

            status = (
                "TEPAT WAKTU"
                if pred == 1
                else "TIDAK TEPAT"
            )

            xai = get_xai_explanation(
                fitur_kelulusan,
                input_data[0],
                model_kelulusan
            )

            cursor.execute("""
            INSERT INTO hasil_prediksi_kelulusan
            (
                nrp,prediksi,probabilitas,
                kategori,xai_json
            )
            VALUES (%s,%s,%s,%s,%s)
            """,(
                nrp,
                int(pred),
                float(prob),
                status,
                json.dumps(xai)
            ))

            hasil.append({
                "nrp": nrp,
                "nama": row["nama"],
                "status": status,
                "probabilitas": round(float(prob),4)
            })

        conn.commit()
        return {
            "message":"Prediksi CSV Kelulusan berhasil",
            "total_data":len(hasil),
            "results":hasil
        }
    finally:
        conn.close()

# CSV PREDICTION - SEMESTER
@app.post("/predict/semester/csv")
async def predict_semester_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="File harus CSV"
        )

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_cols = [
        "nrp","nama","semester_ke","label_prog",
        "ips","ips_diff","presensi","teori",
        "prak","sks_target","delay_tugas",
        "total_sks_hutang","ekonomi_level",
        "skor_masuk"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom tidak lengkap: {missing}"
        )

    conn = get_db()
    cursor = conn.cursor()

    hasil = []

    try:
        for _, row in df.iterrows():
            nrp = str(row["nrp"])

            cursor.execute(
                "SELECT nrp FROM mahasiswa_aktif_semester WHERE nrp=%s",
                (nrp,)
            )

            if cursor.fetchone():
                cursor.execute("""
                UPDATE mahasiswa_aktif_semester SET
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
                """,(
                    row["nama"],
                    row["semester_ke"],
                    row["label_prog"],
                    row["ips"],
                    row["ips_diff"],
                    row["presensi"],
                    row["teori"],
                    row["prak"],
                    row["sks_target"],
                    row["delay_tugas"],
                    row["total_sks_hutang"],
                    row["ekonomi_level"],
                    row["skor_masuk"],
                    nrp
                ))
            else:
                cursor.execute("""
                INSERT INTO mahasiswa_aktif_semester
                (
                    nrp,nama,semester_ke,label_prog,
                    ips,ips_diff,presensi,teori,prak,
                    sks_target,delay_tugas,total_sks_hutang,
                    ekonomi_level,skor_masuk
                )
                VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,(
                    nrp,
                    row["nama"],
                    row["semester_ke"],
                    row["label_prog"],
                    row["ips"],
                    row["ips_diff"],
                    row["presensi"],
                    row["teori"],
                    row["prak"],
                    row["sks_target"],
                    row["delay_tugas"],
                    row["total_sks_hutang"],
                    row["ekonomi_level"],
                    row["skor_masuk"]
                ))

            input_data = [[
                row["semester_ke"],
                row["label_prog"],
                row["ips"],
                row["ips_diff"],
                row["presensi"],
                row["teori"],
                row["prak"],
                row["sks_target"],
                row["delay_tugas"],
                row["total_sks_hutang"],
                row["ekonomi_level"],
                row["skor_masuk"]
            ]]

            X = pd.DataFrame(
                input_data,
                columns=fitur_semester
            )

            X = imputer_semester.transform(X)

            pred = model_semester.predict(X)[0]
            prob = model_semester.predict_proba(X)[0][1]

            status = (
                "AMAN"
                if pred == 1
                else "RISIKO TINGGI"
            )

            xai = get_xai_explanation(
                fitur_semester,
                input_data[0],
                model_semester
            )

            cursor.execute("""
            INSERT INTO hasil_prediksi_semester
            (
                nrp,prediksi,probabilitas,
                kategori,xai_json
            )
            VALUES (%s,%s,%s,%s,%s)
            """,(
                nrp,
                int(pred),
                float(prob),
                status,
                json.dumps(xai)
            ))

            hasil.append({
                "nrp": nrp,
                "nama": row["nama"],
                "status": status,
                "probabilitas": round(float(prob),4)
            })

        conn.commit()
        return {
            "message":"Prediksi CSV Semester berhasil",
            "total_data":len(hasil),
            "results":hasil
        }
    finally:
        conn.close()

# =====================================================
# GENERIC ENDPOINT FOR DETAIL (FIX 404)
# =====================================================

@app.get("/results/{target}/{nrp}")
async def get_result_by_nrp_generic(target: str, nrp: str):
    """
    Generic endpoint untuk detail prediksi
    Mendukung kedua format: /results/kelulusan/{nrp} dan /results/{target}/{nrp}
    """
    if target == "kelulusan":
        return await get_kelulusan_by_nrp(nrp)
    elif target == "semester":
        return await get_result_semester_by_nrp(nrp)
    else:
        raise HTTPException(
            status_code=400,
            detail="Target harus 'kelulusan' atau 'semester'"
        )


# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "server_time": datetime.now()
    }