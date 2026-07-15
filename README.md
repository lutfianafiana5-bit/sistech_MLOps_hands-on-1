# SISTECH 2026 — MLOps Hands-On 1: Risk Score Feature & Label Engineering

**Nama:** Lutfiana Putri Amalia
**Program:** Sisters in Tech by RISTEK Fasilkom UI 2026 - Machine Learning Operations

## 1. Deskripsi Dataset

Dataset yang digunakan adalah **Chicago Crimes** (ekspor 13 Juli 2026), berisi
**231.214 baris** kejadian kejahatan dengan kolom `Date`, `Primary Type`,
`Description`, `Latitude`/`Longitude`, `Arrest`, dll.

Berbeda dari dataset asli Chicago Crimes 2001–sekarang (jutaan baris), file yang
diberikan **sudah berupa subset**: hanya mencakup **6 Juli 2025 – 4 Juli 2026**
(persis ~1 tahun). Karena ukurannya sudah ringan (~230 ribu baris) untuk
dikomputasi, **tidak dilakukan subsetting tambahan** — seluruh data dipakai apa
adanya. Setelah pembersihan (buang baris tanpa koordinat/tanggal valid & filter
bounding box Chicago), tersisa **231.126 baris** (hanya 88 baris dibuang).

## 2. Insight Utama dari EDA

- **Jenis kejahatan**: `THEFT`, `BATTERY`, `CRIMINAL DAMAGE`, `ASSAULT` mendominasi
  (≈61% dari seluruh kejadian).
- **Pola jam**: kejadian meningkat sore–malam (16:00–21:00) dan turun dini hari
  (03:00–06:00). **Catatan kualitas data**: jam `00:00` punya lonjakan tidak wajar
  — kemungkinan artefak pencatatan (waktu tak pasti dibulatkan ke tengah malam),
  bukan pola kriminal asli.
- **Weekday vs weekend**: proporsi kejadian dini hari (00:00–04:00) lebih tinggi
  di akhir pekan; puncak siang hari kerja sedikit lebih tajam.
- **Interaksi hari x jam**: heatmap menunjukkan intensitas bukan sekadar
  penjumlahan efek hari & jam independen — ada interaksi nyata (mis. dini hari
  akhir pekan jauh lebih ramai dari dini hari hari kerja). Ini menjustifikasi unit
  analisis gabungan **(sel x hari x jam)**.
- **Pola spasial**: kejahatan sangat terkonsentrasi di klaster tertentu (hotspot),
  bukan tersebar merata — dasar justifikasi kebutuhan *spatial smoothing*.
- **Arrest rate**: hanya ~15% keseluruhan, bervariasi tajam antar jenis kejahatan;
  sengaja **tidak** dijadikan komponen Risk Score (indikator respons penegak
  hukum, bukan risiko lokasi — berisiko *feedback loop* bila dipakai untuk
  menentukan penempatan patroli), namun disimpan sebagai fitur tambahan opsional.

## 3. Justifikasi Keputusan Desain

### 3.1 Fitur Temporal — Cyclical Encoding
Jam, hari, dan bulan diencode sebagai pasangan `(sin, cos)` dengan periode
masing-masing (24, 7, 12) agar ujung siklus (mis. `23:00` & `00:00`) direpresentasikan
berdekatan secara numerik — dibuktikan lewat perbandingan jarak Euclidean di
notebook. `month_sin/cos` dihitung tapi **tidak dibawa ke unit analisis akhir**
karena unit analisis (sel x hari x jam) sudah meleburkan info bulan, dan cakupan
data 1 tahun (hanya 1 siklus musiman) membuat fitur bulanan riskan overinterpretasi.

### 3.2 Fitur Spasial — Grid Aggregation
Resolusi grid dipilih lewat pengujian empiris kepadatan data, bukan angka
sembarang:

| Step (derajat) | ~Ukuran sel | Jumlah sel | Rata-rata kejadian/sel |
|---|---|---|---|
| 0.010 | ~1.1 km | 711 | 325.1 |
| **0.005** | **~550 m** | **2.450** | **94.3** |
| 0.003 | ~110 m | 6.134 | 37.7 |

Step **0.01** terlalu kasar (resolusi rendah, sel mencampur lingkungan yang
berbeda). Step **0.003** memberi rata-rata <8 kejadian/sel — terlalu jarang begitu
dipecah lagi per (hari x jam), 168 kombinasi waktu/sel. **Step 0.005** dipilih
sebagai titik tengah: skala "lingkungan/beberapa blok", tetap punya kepadatan data
memadai untuk agregasi yang stabil.

### 3.3 Severity Scoring — Rule-Based, Bukan Lookup Table Statis
Dataset punya **332 kombinasi unik** `Primary Type` x `Description`. Lookup table
manual (seperti contoh baseline di dokumen tugas) selalu punya *long tail* yang
jatuh ke satu nilai default generik — bahkan menutup 90% data pun tetap butuh >50
baris tabel, dan 10% sisanya seragam default.

**Keputusan:** severity dibangun sebagai **fungsi aturan**:
1. **Base score per `Primary Type`** (skala 0–100), mengikuti hierarki keparahan ala
   FBI Uniform Crime Report (kejahatan kekerasan terhadap orang > kejahatan
   properti > pelanggaran ketertiban umum), diselaraskan kasar dengan proksi
   klasifikasi felony/misdemeanor KUHP Illinois. Contoh: `HOMICIDE`=100,
   `CRIMINAL SEXUAL ASSAULT`=90, `ROBBERY`=55, `BATTERY`=38 (baseline "simple"),
   `THEFT`=22, `CRIMINAL TRESPASS`=12.
2. **Modifier aditif berbasis kata kunci** di `Description`: `AGGRAVATED` (+15),
   senjata api (+20), pisau/benda tajam (+12), status `ARMED` (+15) vs
   `STRONG ARM`/`NO WEAPON` (−8), elemen `DOMESTIC` (+8), cedera serius (+15) vs
   minor (−6), korban rentan (anak/lansia, +10), status `ATTEMPT`/percobaan (−10).

Pendekatan ini men-generalisasi ke **semua** kombinasi, termasuk yang belum pernah
muncul di data — 100% baris dapat diberi skor yang berjustifikasi tanpa fallback
generik satu-nilai. Validasi kualitatif: `ASSAULT / AGGRAVATED - HANDGUN` (77) >>
`ASSAULT / SIMPLE` (42); `BURGLARY / FORCIBLE ENTRY` (54) >> `CRIMINAL TRESPASS /
TO LAND` (12) — gradasi keparahan yang masuk akal, bukan angka acak.

### 3.4 Temporal Decay — Exponential Half-Life
Peluruhan **eksponensial** dipilih di atas peluruhan linear (versi paling
sederhana) karena: (1) linear membuat kejadian tertua bernilai persis 0
(kehilangan informasi total) dan mengasumsikan laju penurunan konstan; (2)
eksponensial merepresentasikan penurunan **proporsional** (persentase tetap) per
satuan waktu, lebih sesuai intuisi "kejadian yang makin basi makin cepat
kehilangan relevansi lalu melambat".

**Parameter:** half-life = **90 hari** (kejadian berusia 90 hari berkontribusi
separuh dari kejadian hari ini). Dipilih karena untuk kebutuhan operasional
(alokasi patroli/sumber daya), tren kuartal terakhir jauh lebih relevan daripada
kejadian setahun lalu, namun tidak sereaktif window mingguan yang membuat skor
terlalu volatil.

### 3.5 Spatial Decay — Gaussian Kernel via Haversine Distance
Versi paling sederhana (rata-rata seragam 8 tetangga grid 3x3) diperbaiki karena
dua kelemahan: (1) mengasumsikan semua tetangga berkontribusi sama tanpa memandang
jarak sesungguhnya; (2) grid lat/lon **tidak isotropic** (1° longitude di Chicago
≈ 83 km, 1° latitude ≈ 111 km — jarak "kotak" grid tidak merepresentasikan jarak
riil).

**Keputusan:** hitung jarak **great-circle (haversine)** riil antar sel dengan
`BallTree` (scikit-learn), lalu beri bobot **kernel Gaussian** (bandwidth = 0.6 km,
cutoff pencarian 3x bandwidth) — pengaruh tetangga menurun mulus terhadap jarak,
bukan potongan tegas tetangga/bukan-tetangga. Smoothing dilakukan per (hari, jam)
agar pola waktu antar sel tidak tercampur.

### 3.6 Normalisasi
`risk_raw` sangat *skewed* (skewness ≈ 1.97) karena mayoritas kombinasi
sel/hari/jam sepi sementara sedikit hotspot sangat ramai. Min-max langsung akan
didominasi outlier dan menekan mayoritas sel ke ujung bawah skala. Dipakai:
**log1p** (skewness turun ke ≈ −0.31) → **clipping di persentil ke-99** → **min-max
ke 0–100**. Hasil: `risk_score` akhir punya skewness ≈ −0.38, jauh lebih seimbang
dan bisa dibedakan di seluruh rentang.

## 4. Ringkasan Dataset Akhir

`features_labels.csv` / `.parquet` — **138.663 baris x 14 kolom**, unit analisis
(sel grid x hari x jam), tanpa nilai kosong:

| Kolom | Deskripsi |
|---|---|
| `cell_id`, `lat_r`, `lon_r` | Identitas & koordinat pusat sel grid (~550 m) |
| `dow`, `hour` | Hari (0=Senin) & jam mentah |
| `dow_sin/cos`, `hour_sin/cos` | Encoding siklikal |
| `crime_count` | Frekuensi kejadian di unit tsb |
| `arrest_rate` | Rata-rata status penangkapan (fitur tambahan, bukan label) |
| `base_value` | Σ(severity x bobot recency) sebelum spatial smoothing |
| `risk_raw` | `base_value` setelah spatial smoothing (Gaussian/haversine) |
| **`risk_score`** | **Target label 0–100** setelah log1p + clip P99 + min-max |

Distribusi `risk_score`: mean 61.6, median 63.1, std 18.2, rentang 0–100 penuh
terpakai.

## 5. Refleksi (Kendala & Solusi)

- **Long-tail kombinasi Primary Type x Description** → diatasi dengan severity
  berbasis aturan (bukan lookup manual), sehingga men-generalisasi ke kombinasi
  yang belum terlihat.
- **Cakupan data hanya 1 tahun** → fitur bulanan sengaja tidak dibawa ke unit
  analisis akhir; dicatat sebagai keterbatasan, bukan disembunyikan.
- **Grid lat/lon tidak isotropic** → diatasi dengan jarak haversine riil (bukan
  Euclidean grid) saat spatial smoothing.
- **`risk_raw` sangat skewed** → dicek langsung lewat skewness & histogram,
  diatasi dengan log1p + percentile clipping sebelum min-max.
- **Keterbatasan yang disadari**: severity table berbasis aturan tetap bersifat
  *judgment call* penulis (bukan hasil studi/expert panel formal) — cukup untuk
  MVP hands-on ini, tapi perlu divalidasi lebih lanjut sebelum dipakai untuk
  keputusan operasional nyata. `arrest_rate` berpotensi bias (area dengan
  pengawasan lebih intensif bisa tercatat arrest rate lebih tinggi tanpa berarti
  lebih berbahaya) dan perlu kehati-hatian bila dipakai sebagai fitur model di
  Hands-On 2.

## 6. Referensi
- [Chicago Crimes 2001–Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
- pandas — Time series / date functionality
- scikit-learn — Preprocessing & feature encoding
- scikit-learn — Nearest Neighbors (BallTree, haversine metric)
- FBI Uniform Crime Reporting (UCR) Program — Part I/Part II offense hierarchy
