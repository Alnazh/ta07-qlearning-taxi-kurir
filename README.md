# TA-07: Q-Learning pada Agen Kurir Pengiriman Paket (Taxi Problem)

Tugas Akhir 07 mata praktikum **Pembelajaran Mesin** (Modul 07: Reinforcement Learning).
Dosen pengampu: Mohammad Bayu Anggara, S.Kom., M.Kom.

Proyek ini mengimplementasikan **Q-Learning tabular** pada masalah kurir yang harus menjemput paket
lalu mengantarkannya di grid kota 5×5. Lingkungan dibangun sendiri dengan Python murni (tanpa Gymnasium),
lalu dianalisis dari sisi kurva pembelajaran, kebijakan yang dipelajari, dan pengaruh *discount factor* (γ).

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](`https://colab.research.google.com/drive/1KfW7r00SOzYcYz68B-QV01ELmiEBT-SE?usp=sharing`)

---

## Skenario

Seorang kurir bergerak di grid 5×5 dengan empat titik khusus: **R** (0,0), **G** (0,4), **Y** (4,0), dan **B** (4,4).
Pada awal setiap episode, posisi kurir, lokasi paket, dan tujuan diacak. Kurir harus menuju paket,
mengambilnya, lalu mengantarnya ke tujuan dengan langkah sesedikit mungkin.

| Elemen | Definisi |
|---|---|
| **State (S)** | (baris kurir, kolom kurir, lokasi paket 0–4, tujuan 0–3) = 5×5×5×4 = **500 state** (400 yang dapat dikunjungi) |
| **Action (A)** | 6 aksi: Selatan, Utara, Timur, Barat, Ambil, Antar |
| **Reward (R)** | −1 per langkah; +20 jika paket diantar ke tujuan yang benar (terminal); −10 jika Ambil/Antar dilakukan di tempat yang salah |

Lokasi paket bernilai 0–3 jika paket menunggu di R/G/Y/B, dan 4 jika sudah berada di dalam kendaraan.

## Hasil Ringkas

Hasil berikut berasal dari pelatihan utama (seed 42, 5.000 episode, α = 0,8, γ = 0,9).

| Metrik | Hasil |
|---|---|
| Rata-rata reward 100 episode pertama | −690,12 |
| Rata-rata reward 100 episode terakhir | 8,72 |
| Konvergensi (kurva mendatar) | sekitar episode ke-3.474 |
| Keberhasilan kebijakan *greedy* (200 episode baru) | 100% |
| Keberhasilan dari seluruh 400 state awal | 400 dari 400 |
| Rute dengan langkah minimum | 394 dari 400 (98,5%) |

Eksperimen γ (rata-rata ± simpangan baku pada 10 seed):

| | γ = 0,9 | γ = 0,1 | γ = 0,01 |
|---|---|---|---|
| Reward 100 episode akhir | 8,74 ± 0,71 | 8,79 ± 0,54 | −39,99 ± 10,64 |
| Konvergensi (episode) | 3.362 ± 238 | 3.386 ± 132 | 4.994 ± 8 |
| Sukses *greedy* | 100% | 100% | 37,5% |
| State awal yang berhasil | 100% | 99,98% | 55,0% |

γ = 0,01 membuat agen "miopik" dan gagal terutama pada fase menuju paket. γ = 0,1 ternyata masih berhasil
karena lingkungannya deterministik.

## Struktur Repositori

```
.
├── TA07_Taxi_QLearning.ipynb          # Notebook Colab: Langkah 1-5 (jalankan dari atas ke bawah)
├── ta07_taxi_qlearning.py             # Lingkungan, Q-Learning, kurva, kebijakan, evaluasi (Langkah 1-4)
├── ta07_eksperimen_hiperparameter.py  # Eksperimen γ pada seed 42 (Langkah 5)
├── ta07_multi_seed.py                 # Pengulangan 10 seed + keoptimalan rute
├── ta07_cek_klaim_laporan.py          # Mencocokkan angka di laporan dengan hasil kode
├── ta07_multi_seed_hasil.csv          # Hasil mentah uji 10 seed
├── requirements.txt
├── .gitignore
├── docs/                              # Laporan TA-07 (DOCX dan PDF)
└── output/                            # Dibuat otomatis: gambar, q_table.npy, CSV
```

## Cara Menjalankan

### Opsi 1: Google Colab (paling mudah)
1. Buka `TA07_Taxi_QLearning.ipynb` di Colab (tombol di atas).
2. Pilih **Runtime → Run all**.

### Opsi 2: Lokal
```bash
git clone `[tempel URL repositori]`
cd `[nama-repositori]`
pip install -r requirements.txt

python ta07_taxi_qlearning.py              # Langkah 1-4, gambar tersimpan di output/
python ta07_eksperimen_hiperparameter.py   # Langkah 5 (butuh berkas di atas dalam folder yang sama)
python ta07_multi_seed.py                  # Uji 10 seed (membutuhkan waktu beberapa menit)
python ta07_cek_klaim_laporan.py           # Memeriksa kesesuaian angka laporan dengan hasil kode
```

Untuk eksperimen *learning rate*, ubah `PARAM = "alpha"` di `ta07_eksperimen_hiperparameter.py`.

## Hiperparameter Dasar

| Parameter | Nilai |
|---|---|
| α (learning rate) | 0,8 |
| γ (discount factor) | 0,9 |
| Episode | 5.000 |
| Langkah maksimum per episode | 200 |
| ε awal / minimum | 1,0 / 0,01 |
| Laju penurunan ε | 0,001 (eksponensial) |
| Seed | 42 (utama), 0–9 (uji kekokohan) |

## Verifikasi Laporan

`ta07_cek_klaim_laporan.py` menghitung ulang angka-angka yang dikutip di laporan 
(pelatihan utama, tabel titik pelatihan, evaluasi kebijakan, eksperimen γ, dan ringkasan 10 seed), lalu
menampilkan tabel "di laporan" versus "dari kode" beserta status cocok atau tidak. Jalankan setelah
`ta07_multi_seed.py`. Skrip keluar dengan kode 1 jika ada angka yang tidak cocok.

## Catatan

- Hasil dengan seed yang sama seharusnya identik di mesin lain, selama versi NumPy dan pustaka lain tidak
  berbeda jauh.
- Lingkungan bersifat deterministik dan tanpa dinding, jadi hasil belum tentu berlaku pada lingkungan yang
  lebih rumit atau stokastik.
- Penjelasan sebab-akibat tentang kegagalan γ = 0,01 pada fase menuju paket adalah interpretasi, bukan hasil
  pengujian.

## Referensi

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning, 8*(3–4), 279–292.
- Dietterich, T. G. (2000). Hierarchical reinforcement learning with the MAXQ value function decomposition.
  *Journal of Artificial Intelligence Research, 13*, 227–303.

