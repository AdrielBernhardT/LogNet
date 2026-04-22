# LogNet — Anomaly Detection & Attack Classification dari Log File menggunakan Temporal Graph Network

## Deskripsi

**LogNet** adalah sistem deteksi anomali dan klasifikasi serangan berbasis **Temporal Graph Network (TGN)** yang bekerja pada data log autentikasi jaringan. Sistem ini mampu:

- **Mendeteksi** aktivitas mencurigakan yang menyimpang dari pola normal
- **Mengklasifikasi** jenis serangan yang terjadi ke dalam tiga kategori: Reconnaissance, Lateral Movement, dan Persistence

### Dataset

| File | Keterangan |
|---|---|
| `auth.txt.gz` | Log autentikasi seluruh jaringan (~1.6 miliar baris) |
| `flows.txt.gz` | Data aliran jaringan |
| `redteam.txt.gz` | Ground truth serangan red team (702 events) |

Sumber dataset: **LANL (Los Alamos National Laboratory) Unified Host and Network Dataset**

---

## Arsitektur Model

```
Log Events (Edge Stream)
        │
        ▼
  MemoryModule          ← Menyimpan state tiap node (GRU-based)
        │
        ▼
  MessageModule         ← Menghitung pesan antar node dari edge
        │
        ▼
  EmbeddingModule       ← GraphSAGE untuk embedding node aktif
        │
        ▼
  ┌─────────────────┐
  │  Anomaly Head   │   → Skor anomali (0–1) per edge
  │  Class Head     │   → Kelas serangan (Recon / Lateral / Persistence)
  └─────────────────┘
```

---

## Alur Pipeline

```
Cell 1  Install dependencies
Cell 2  Import library
Cell 3  Konfigurasi path & konstanta
Cell 4  Cek ketersediaan file dataset
   │
   ▼
Cell 5  Bangun node map (scan semua user & komputer unik)
Cell 6  Bersihkan & muat data redteam sebagai ground truth
Cell 7  Analisis distribusi attack event & daftar attacker
Cell 8  Analisis pola serangan tiap attacker
Cell 9  Visualisasi distribusi serangan
Cell 10 Profil attacker & penetapan kelas serangan
Cell 11 Bangun lookup tabel attack_class per event
   │
   ▼
Cell 12 Reload semua variabel (node_map, redteam_lookup, class_lookup)
Cell 13 Proses auth.txt.gz → edge parquet (labeling + encoding fitur)
Cell 14 Merge semua batch parquet menjadi file final
Cell 15 Cek isi folder ./parsed/
Cell 16 Cek working directory & lokasi file
Cell 17 Verifikasi statistik label dari file final
Cell 18 Verifikasi konsistensi label vs attack_class
Cell 19 Cek skema kolom parquet & distribusi label
   │
   ▼
Cell 20 Konfigurasi split train/val/test + konstanta model
Cell 21 Definisi arsitektur TGN (Memory, Message, Embedding, Head)
Cell 22 Inisialisasi model, loss function, optimizer
Cell 23 Stratified temporal sampling (hemat RAM)
Cell 24 Pre-index row groups untuk fast windowed loading
Cell 25 Fungsi load_window_fast (generator per time window)
Cell 26 Benchmark estimasi waktu training
Cell 27 Analisis representasi sampling per split
Cell 28 Analisis distribusi attack per hari
Cell 29 Training loop (dengan early stopping)
Cell 30 Evaluasi pada test set + visualisasi metrik
Cell 31 Simpan model final + fungsi inference
```

---

## Penjelasan Tiap Cell

### Cell 1 — Installation

Menginstall seluruh dependensi Python yang dibutuhkan:

| Library | Fungsi |
|---|---|
| `pandas`, `numpy` | Manipulasi data tabular |
| `torch`, `torch-geometric` | Framework deep learning & graph neural network |
| `scikit-learn` | Metrik evaluasi & preprocessing |
| `matplotlib`, `seaborn` | Visualisasi |
| `dask`, `pyarrow` | Pemrosesan file besar secara efisien |
| `torch-scatter`, `torch-sparse` | Operasi sparse tensor untuk PyG |

---

### Cell 2 — Import

Mengimpor semua library ke dalam namespace. Juga mengonfigurasi logging dengan format timestamp dan level INFO, serta mencetak versi PyTorch, PyG, dan device yang aktif (CPU/CUDA).

---

### Cell 3 — Configuration

Mendefinisikan semua konstanta global:

| Variabel | Nilai | Keterangan |
|---|---|---|
| `AUTH_PATH` | path ke `auth.txt.gz` | File log autentikasi utama |
| `REDTEAM_PATH` | path ke `redteam.txt.gz` | Ground truth serangan |
| `OUTPUT_DIR` | `./parsed` | Folder output semua file intermediate |
| `CHUNKSIZE` | 500.000 | Jumlah baris per chunk saat membaca CSV |
| `AUTH_TYPE_MAP` | dict | Encoding tipe autentikasi (NTLM, Kerberos, dll) |
| `LOGON_TYPE_MAP` | dict | Encoding tipe logon (Network, Interactive, dll) |
| `AUTH_ORIENT_MAP` | dict | Encoding orientasi auth (LogOn, LogOff, TGT, dll) |

---

### Cell 4 — Path and Size Check

Memverifikasi bahwa kedua file dataset (`auth.txt.gz` dan `redteam.txt.gz`) benar-benar ada di path yang dikonfigurasi, sekaligus menampilkan ukuran file dalam GB. Ini mencegah error yang membingungkan di cell berikutnya jika file tidak ditemukan.

---

### Cell 5 — Build Node Map

Melakukan **scan penuh** terhadap seluruh file `auth.txt.gz` untuk mengumpulkan semua entitas unik (user dan komputer) yang pernah muncul dalam log.

**Proses:**
1. Baca file per chunk 500.000 baris
2. Kumpulkan semua nilai unik dari kolom `src_user`, `dst_user`, `src_computer`, `dst_computer`
3. Assign integer ID unik untuk setiap entitas:
   - User diberi prefix `u_` → ID 0, 1, 2, ...
   - Komputer diberi prefix `c_` → ID dilanjut dari user
4. Simpan ke `./parsed/node_map.parquet`

**Output:** Dictionary `node_map` dengan format `{"u_UserA": 0, "c_Comp1": 1234, ...}`

> ⚠️ Proses ini membutuhkan waktu ~30–60 menit untuk dataset penuh 1.6 miliar baris.

---

### Cell 6 — Cleaning Attack Dataset

Memuat file `redteam.txt.gz` dan membuat **lookup set** untuk ground truth serangan.

**Proses:**
1. Baca `redteam.txt.gz` dengan kolom: `time`, `src_user`, `src_computer`, `dst_computer`
2. Buat set dari tuple `(time, src_user, src_computer, dst_computer)` sebagai kunci unik
3. Cek duplikat dan nilai NaN untuk memahami perbedaan antara 748 baris raw vs 715 events unik

**Output:** `redteam_lookup` — set yang digunakan untuk memberi label `attack=1` saat memproses auth.

---

### Cell 7 — Checking Attack Event and Attacker List

Analisis deskriptif dasar terhadap data redteam:

- Total attack events setelah deduplication
- Jumlah attacker unik
- Jumlah komputer sumber dan tujuan yang terlibat
- Daftar attacker diurutkan berdasarkan jumlah serangan

---

### Cell 8 — Check Attack Pattern

Menganalisis pola serangan tiap attacker secara individual:

| Metrik | Keterangan |
|---|---|
| `Events` | Total event serangan |
| `Duration` | Rentang waktu serangan (detik & jam) |
| `Src comp` | Jumlah mesin sumber berbeda yang digunakan |
| `Dst comp` | Jumlah target berbeda yang diserang |
| `Time range` | Timestamp awal dan akhir serangan |

Informasi ini menjadi dasar untuk klasifikasi jenis serangan di Cell 10.

---

### Cell 9 — Visualize Plot

Menghasilkan dua plot analisis redteam yang disimpan ke `./parsed/redteam_analysis.png`:

1. **Attack events per hari per attacker** — Bar chart yang menunjukkan kapan dan seberapa aktif tiap attacker
2. **Jumlah komputer unik per attacker** — Horizontal bar chart yang menunjukkan luas jangkauan serangan

---

### Cell 10 — Attacker Profile and Classification Parameter

Membangun **profil tiap attacker** dan menetapkan kelas serangan berdasarkan aturan heuristik:

| Kondisi | Kelas | Nama |
|---|---|---|
| Duration > 100 jam **DAN** target ≥ 5 | `2` | Lateral Movement |
| Duration > 20 jam | `3` | Persistence |
| Selain itu | `1` | Reconnaissance |

Profil disimpan ke `./parsed/attacker_profiles.parquet`.

> Logika ini diprioritaskan eksplisit — Lateral Movement dicek lebih dulu sebelum Persistence untuk menghindari misklasifikasi.

---

### Cell 11 — Attack Classification

Membangun `redteam_class_lookup` — dictionary dengan kunci tuple `(time, src_user, src_computer, dst_computer)` dan nilai berupa kelas serangan (1, 2, atau 3).

Dictionary ini digunakan di Cell 13 untuk memberi label `attack_class` pada setiap event serangan yang ditemukan saat memproses auth.

---

### Cell 12 — Variable Check

Cell **reload** semua variabel penting dari file yang sudah disimpan:

- `node_map` ← `node_map.parquet`
- `redteam_lookup` ← re-parse `redteam.txt.gz`
- `redteam_class_lookup` ← rebuild dari `attacker_profiles.parquet`

Berguna jika kernel restart setelah Cell 5–11 selesai berjalan, sehingga tidak perlu mengulang proses yang panjang dari awal.

---

### Cell 13 — Auth Process

**Cell terpanjang dan terpenting** — memproses seluruh `auth.txt.gz` menjadi file parquet siap training.

**Proses per chunk:**
1. Hapus baris dengan `src_computer` atau `dst_computer` kosong
2. Konversi timestamp ke integer
3. Map nama komputer dan user ke integer ID via `node_map`
4. Bangun kunci `time|src_user|src_computer|dst_computer` untuk pencocokan redteam
5. Assign `label = 1` jika kunci ada di `redteam_lookup`, else `0`
6. Assign `attack_class` dari `redteam_class_lookup` (0=normal, 1/2/3=kelas serangan)
7. Encode 6 fitur numerik via `encode_features()`
8. Tulis ke batch parquet setiap 10 chunk

**6 Fitur yang di-encode:**

| Index | Fitur | Keterangan |
|---|---|---|
| 0 | `feat_auth_type` | Tipe autentikasi (NTLM=0, Kerberos=1, dll) |
| 1 | `feat_logon_type` | Tipe logon (Network=0, Interactive=1, dll) |
| 2 | `feat_auth_orient` | Orientasi (LogOn=0, LogOff=1, TGT=2, dll) |
| 3 | `feat_success` | Sukses=1.0, Gagal=0.0 |
| 4 | `feat_hour` | Jam kejadian (0–24) |
| 5 | `feat_offhours` | 1.0 jika terjadi di luar jam kerja (sebelum 08:00 atau setelah 18:00) |

**Output:** File batch `edges_batch_XXXX.parquet` dan `feats_batch_XXXX.parquet` di `./parsed/`

---

### Cell 14 — Merge

Menggabungkan seluruh batch parquet menjadi dua file final:

- `./parsed/auth_edges.parquet` — informasi edge (src_id, dst_id, timestamp, label, attack_class)
- `./parsed/auth_feats.parquet` — 6 fitur numerik per edge

**Proses:**
1. Baca tiap pasang batch edge + feat
2. Urutkan berdasarkan timestamp (penting untuk TGN yang temporal)
3. Tulis ke file final menggunakan `ParquetWriter` (streaming, tidak load semua ke RAM)
4. Hapus file batch setelah merge selesai

---

### Cell 15 — Check Parsed Folder

Menampilkan daftar semua file di `./parsed/` beserta ukurannya dalam MB. Verifikasi cepat bahwa semua file output terbentuk dengan benar.

---

### Cell 16 — Checking File Path

Melakukan `os.walk` untuk menemukan semua file yang mengandung kata kunci `batch`, `edges`, `feats`, atau `node_map` di seluruh direktori kerja. Berguna untuk debugging jika file tersimpan di lokasi yang tidak terduga.

---

### Cell 17 — Categorizing Attack

Membaca `auth_edges.parquet` secara streaming (5 juta baris per batch) untuk menghitung statistik dasar tanpa memuat seluruh file ke RAM:

- Total edges
- Jumlah attack events (label=1) vs normal (label=0)
- Attack ratio
- Rentang timestamp (ts_min dan ts_max)

---

### Cell 18 — Checking Edges

Memverifikasi **konsistensi** antara kolom `label` dan `attack_class`:

| Kondisi yang diverifikasi | Harapan |
|---|---|
| `label==1` → `attack_class > 0` | Semua attack harus punya kelas serangan |
| `label==0` → `attack_class == 0` | Normal tidak boleh punya kelas serangan |

Jika ditemukan inkonsistensi, cell ini akan memperingatkan untuk re-run Cell 13.

---

### Cell 19 — Checking Distribution Attack and Label

Menampilkan skema kolom lengkap dari kedua file parquet dan distribusi aktual `label` serta `attack_class` dari 10 juta baris pertama sebagai sample.

---

### Cell 20 — Load Split Config + Konstanta

Menentukan **batas waktu** untuk split train/val/test berdasarkan timestamp aktual dataset:

| Split | Rentang | Jumlah Hari |
|---|---|---|
| Train | hari 0 – 6 | 7 hari |
| Val | hari 7 – 14 | 8 hari |
| Test | hari 15 – 57 | ~43 hari |

Juga mendefinisikan semua konstanta model:

| Konstanta | Nilai | Keterangan |
|---|---|---|
| `MEMORY_DIM` | 64 | Dimensi vektor memori tiap node |
| `EMBED_DIM` | 64 | Dimensi embedding output |
| `EDGE_FEAT_DIM` | 6 | Jumlah fitur per edge |
| `NUM_CLASSES` | 3 | Recon, Lateral, Persistence |
| `WINDOW_SIZE` | 86.400 | 1 hari dalam detik |
| `BATCH_SIZE` | 2.048 | Event per mini-batch |
| `LR` | 1e-3 | Learning rate Adam |

---

### Cell 21 — Temporal Graph Network Architecture

Mendefinisikan arsitektur TGN yang terdiri dari 4 komponen:

#### MemoryModule
Menyimpan representasi historis tiap node menggunakan **GRU Cell**. Memory diupdate setiap kali sebuah node terlibat dalam event baru.

```
Memory[node] = GRU(message, Memory[node])
```

#### MessageModule
Menghitung "pesan" dari sebuah edge dengan menggabungkan:
- Memory node sumber
- Memory node tujuan
- 6 fitur edge
- Delta waktu sejak event terakhir node sumber

#### EmbeddingModule
Menggunakan **GraphSAGE** untuk mengagregasi informasi dari tetangga node, menghasilkan embedding kontekstual. Hanya memproses node yang aktif dalam batch saat ini untuk efisiensi.

#### TGN (Model Utama)
Menggabungkan semua komponen di atas dan memiliki dua output head:

- **Anomaly Head** — Binary classification (normal vs attack), output skor 0–1
- **Class Head** — Multi-class classification (Recon / Lateral / Persistence), hanya aktif untuk attack events

---

### Cell 22 — Initialize Model

Membuat instance model TGN dan mendefinisikan komponen training:

**Loss Functions:**

| Loss | Digunakan untuk | Keterangan |
|---|---|---|
| `FocalLoss` | Anomaly detection | α=0.75, γ=2.0 — mengatasi class imbalance ekstrem (attack ratio ~0.00004%) |
| `CrossEntropyLoss` | Attack classification | Weight [10, 10, 10] untuk ketiga kelas |

**Optimizer:** Adam dengan `ReduceLROnPlateau` scheduler (patience=3, factor=0.5)

---

### Cell 23 — Stratified Temporal Sampling

Karena dataset sangat besar (~1.6 miliar event normal vs ~702 attack), dilakukan **sampling adaptif** per split untuk membuat dataset training yang manageable:

| Split | Ratio Normal yang Diambil |
|---|---|
| Train (hari 0–6) | 10% |
| Val (hari 7–14) | 5% |
| Test (hari 15–57) | 2% |

Sampling dilakukan **per hari** agar distribusi temporal terjaga. Semua attack events (702) selalu disertakan 100%.

**Output:** `auth_edges_sampled.parquet` dan `auth_feats_sampled.parquet`

---

### Cell 24 — Pre-Index Training

Membangun **indeks row group** dari file parquet yang sudah di-sample. Untuk setiap row group, dicatat:

- Nomor row group
- Jumlah baris
- Timestamp minimum dan maksimum

Indeks ini disimpan ke `./parsed/batch_index.parquet` dan digunakan oleh `load_window_fast` di Cell 25 untuk membaca hanya row group yang relevan dengan time window tertentu, tanpa scan seluruh file.

---

### Cell 25 — Load Window

Mendefinisikan **generator** `load_window_fast(ts_start, ts_end)` yang:

1. Menggunakan `batch_index_df` untuk menemukan row group yang overlap dengan window waktu
2. Hanya membaca row group yang relevan dari disk
3. Mem-filter baris di luar window
4. Membagi menjadi mini-batch `BATCH_SIZE`
5. Melakukan shift `attack_class`: nilai 1/2/3 → 0/1/2 agar sesuai dengan indeks `CrossEntropyLoss`
6. Yield tuple `(src, dst, feat, ts, label, attack_class, edge_index)` sebagai tensor siap GPU

---

### Cell 26 — Training Estimation Loop

Menjalankan **5 batch pertama** dari training window untuk mengukur kecepatan rata-rata per batch, lalu memproyeksikan:

- Estimasi waktu per epoch
- Estimasi total waktu untuk 10 epoch
- Estimasi dengan early stopping ~4 epoch

Berguna untuk memperkirakan apakah training bisa selesai dalam waktu yang tersedia.

---

### Cell 27 — Train Split Attack Check

Analisis mendalam representasi sampling:

1. **Distribusi temporal** — Berapa baris per hari per split, apakah ada hari yang tidak terwakili
2. **Distribusi fitur** — Perbandingan distribusi jam kejadian antara dataset original vs sampled (memastikan sampling tidak bias temporal)
3. **Coverage attack** — Berapa persen dari 702 attack events yang masuk ke sampled dataset

Menghasilkan tiga plot yang disimpan ke `./parsed/sampling_analysis.png`.

---

### Cell 28 — Distribution Attack Check

Melakukan scan pada `auth_edges.parquet` (full, bukan sampled) untuk mengetahui distribusi attack events per hari kalender. Output menunjukkan:

- Hari mana saja yang mengandung attack
- Hari pertama dan terakhir terjadi serangan
- Perbandingan dengan batas split `TS_TRAIN_END` dan `TS_VAL_END`

Informasi ini memvalidasi apakah strategi split train/val/test sudah mengandung cukup attack events di tiap split.

---

### Cell 29 — Training Loop

**Loop training utama** dengan mekanisme:

- **Early stopping** — berhenti jika val loss tidak membaik selama `PATIENCE=3` epoch
- **Memory reset** — di awal setiap epoch train, memory node direset agar tidak ada kebocoran informasi antar epoch
- **Gradient clipping** — `max_norm=1.0` untuk stabilitas training
- **Loss gabungan** — `loss = loss_anomaly + 0.5 * loss_classification`
- **Checkpoint** — model terbaik disimpan ke `./parsed/best_model.pt` berdasarkan val loss terendah

**Metrik yang dilacak per epoch:**

| Metrik | Keterangan |
|---|---|
| `loss` | Total loss gabungan |
| `loss_ano` | Focal loss anomaly detection |
| `loss_cls` | CrossEntropy loss klasifikasi |
| `auroc` | Area Under ROC Curve |
| `f1_ano` | F1-score anomaly detection |
| `prec_ano` | Precision anomaly detection |
| `rec_ano` | Recall anomaly detection |
| `f1_cls` | Macro F1-score klasifikasi serangan |

---

### Cell 30 — Evaluation on Test Set

Evaluasi komprehensif pada test set (hari 15–57) menggunakan best model:

**Alur evaluasi:**
1. Load best model dari checkpoint
2. **Warm-up memory** dengan val set — agar model punya konteks historis sebelum mengevaluasi test set (menghindari cold-start)
3. Inference pada seluruh test set tanpa gradient
4. **Threshold search** — mencari threshold optimal (0.1–0.9) yang memaksimalkan F1-score
5. Simpan `best_thresh` ke checkpoint

**Output:**
- Classification report anomaly detection (Normal vs Attack)
- Classification report attack classification (Recon / Lateral / Persistence)
- Tiga plot disimpan ke `./parsed/evaluation_results.png`:
  - ROC Curve dengan AUC
  - Precision-Recall Curve
  - Confusion Matrix klasifikasi serangan

---

### Cell 31 — Save & Inference

**Menyimpan model final** ke `./parsed/lognet_final.pt` beserta konfigurasi dan history training.

**Mendefinisikan fungsi `predict_log_batch()`** untuk inference pada data baru:

```python
result_df = predict_log_batch(
    src_computers = ["C1234", "C5678"],
    dst_computers = ["C9999", "C1111"],
    features      = np.zeros((2, 6), dtype=np.float32),
    timestamps    = [1234567, 1234568],
    threshold     = 0.5,
)
```

**Output DataFrame:**

| Kolom | Keterangan |
|---|---|
| `src_computer` | Nama komputer sumber |
| `dst_computer` | Nama komputer tujuan |
| `timestamp` | Unix timestamp event |
| `anomaly_score` | Skor anomali (0.0–1.0) |
| `is_anomaly` | 1 jika terdeteksi serangan |
| `attack_class` | 0=Recon, 1=Lateral, 2=Persistence |
| `attack_class_name` | Label teks kelas serangan |

---

## Struktur Output Folder

```
./parsed/
├── node_map.parquet           # Mapping nama node → integer ID
├── attacker_profiles.parquet  # Profil & kelas tiap attacker
├── split_config.parquet       # Batas timestamp train/val/test
├── batch_index.parquet        # Indeks row group untuk fast loading
├── auth_edges.parquet         # Seluruh edge dengan label (full)
├── auth_feats.parquet         # Seluruh fitur edge (full)
├── auth_edges_sampled.parquet # Edge setelah stratified sampling
├── auth_feats_sampled.parquet # Fitur setelah stratified sampling
├── best_model.pt              # Checkpoint model terbaik (val loss)
├── lognet_final.pt            # Model final + config + history
├── redteam_analysis.png       # Visualisasi pola serangan redteam
├── sampling_analysis.png      # Analisis representasi sampling
└── evaluation_results.png     # ROC, PR Curve, Confusion Matrix
```

---

## Kelas Serangan

| Kode | Nama | Ciri Khas |
|---|---|---|
| 1 | **Reconnaissance** | Serangan singkat, sedikit target, awal kampanye |
| 2 | **Lateral Movement** | Durasi panjang (>100 jam), banyak target (≥5 komputer) |
| 3 | **Persistence** | Durasi sedang-panjang (>20 jam), pola berkelanjutan |

---

## Referensi

- [LANL Dataset](https://csr.lanl.gov/data/cyber1/)
- [Temporal Graph Networks (Rossi et al., 2020)](https://arxiv.org/abs/2006.10637)
- [GraphSAGE (Hamilton et al., 2017)](https://arxiv.org/abs/1706.02216)
- [Focal Loss (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
