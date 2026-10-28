# Catatan digitisasi v2 (Data-EKG-update)

Ditulis 6 Oktober 2026.

## Ringkasnya

Pipeline baru untuk halaman tempelan strip MAC 400 di Data-EKG-update. Satu gambar halaman masuk, keluar 12 lead dalam
mV pada 500 Hz, label kualitas per lead, dan gambar overlay untuk dicek mata. Jalan di CPU, tanpa training, tanpa GPU.
Format keluarannya sama dengan kontrak record di `ecg-classification-pjb_normal` (`signal` 12 x 2000, `ok`, `present`).

## Kenapa tidak meneruskan pipeline sebelumnya

Pipeline `ecg-digitization-synthetic_realistic` mencari panel dari teks footer dan disetel pada 39 scan
`ecg-mac400-scan`. Halaman Data-EKG-update bentuknya kolase: 4 sampai 8 potongan strip ditempel, sebagian diputar 90
derajat, miring, saling menimpa, ada percobaan ulang, ada strip laporan. Pipeline v2 berangkat dari hal yang selalu ada
di tiap strip, apa pun susunan tempelannya: kotak hitam di header MAC 400. Dari kotak itu posisi baris, pulsa kalibrasi,
label lead, dan footer bisa diturunkan.

## Hasil uji

Uji akurasi pada 30 halaman sintetis bergaya MAC 400 yang sinyal aslinya diketahui (133 strip, 389 lead):

| Ukuran | Nilai |
|---|---|
| Strip ditemukan | 132 dari 133 |
| Grup lead benar / tidak terbaca / salah | 89,4 % / 9,8 % / 0,8 % |
| Gain benar | 100 % |
| Korelasi Pearson per lead (median) | 0,93 |
| SNR per lead (median) | 8,5 dB |
| RMSE per lead (median) | 0,10 mV |

Hasil pada seluruh Data-EKG-update (1151 gambar, dijalankan di laptop Windows 6 Oktober 2026, versi library tercatat di `requirements-lock.txt`, detailnya di `digitized_v2_win\RESULTS.md`):

| Ukuran | Nilai |
|---|---|
| Halaman EKG / formulir diagnosis / format lain | 577 / 570 / 4 |
| Strip ditemukan di halaman EKG | 3012 |
| Halaman EKG dengan 12 lead berstatus ok | 226 (39 %) |
| Halaman EKG dengan minimal 9 lead ok | 469 (81 %) |
| Strip yang tidak berhasil dinamai | 265 dari 3012 |
| Gain yang diasumsikan 10 mm/mV | 337 strip |

Data asli tidak punya ground truth. Korelasi Einthoven (II = I + III) di strip limb mediannya 0,75. Angka ini bukan skor
akurasi langsung: di halaman sintetis, strip limb yang lead-nya berkorelasi 0,93 dengan sinyal asli juga hanya mencapai
sekitar 0,8, karena identitas itu menjumlahkan galat tiga lead. Jadi hasil di data asli ada di kisaran yang sama dengan
uji sintetis, tapi tetap perlu dicek mata lewat overlay.

## Cara menjalankan di Windows (PowerShell)

Sekali saja, untuk menyiapkan:

```
cd "D:\Telkom University\Riset\Riset ECG\Digitalisasi dataset\ecg-digitization-v2"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
python -m pytest tests -q
```

Menjalankan seluruh dataset (bisa dihentikan dan dilanjutkan, halaman yang sudah selesai dilewati):

```
cd "D:\Telkom University\Riset\Riset ECG\Digitalisasi dataset\ecg-digitization-v2"
.\.venv\Scripts\Activate.ps1
python -m ecgdig.run --input "..\Data-EKG-update" --out "..\digitized_v2_win" --workers 4
python tools\pair_pages.py --run "..\digitized_v2_win"
python tools\summarize_run.py --run "..\digitized_v2_win"
```

Ekspor ke format repo klasifikasi (setelah kolom label diisi di `pairs.csv`, kolom `ecg_file` dan `label`):

```
python tools\to_classification_index.py --run "..\digitized_v2_win" --out "..\digitized_v2_win\classification" --labels "..\digitized_v2_win\pairs.csv" --min-ok 9
```

## Yang perlu diketahui saat memakai hasilnya

1. Tiga lead dalam satu strip direkam bersamaan, tapi empat strip (I-III, aVR-aVF, V1-V3, V4-V6) adalah empat rekaman
   berbeda, masing-masing sekitar 2,9 detik. Di `record.npz` tiap strip dimulai dari t = 0, jadi 12 lead tidak sejajar
   per denyut.
2. Lead berstatus `ok` saja yang sebaiknya dipakai untuk latih atau uji. `low_quality` biasanya artefak gerak (anak
   menangis) atau jalur yang melompat; cek `overlay.jpg`.
3. `name_source = page_completion` artinya nama grup lead ditebak dari grup yang belum ada di halaman itu. Lebih tidak
   pasti dibanding `glyphs`.
4. Gain `assumed` artinya pulsa kalibrasi terpotong dan teks footer tidak terbaca, jadi dipakai 10 mm/mV.
5. Halaman format lain (lembar EDAN dan lembar ritme panjang) ditandai `ecg_other_layout` dan belum didigitisasi.

## Temuan soal label

Di README repo klasifikasi dan di formulir diagnosis, PJB berarti Penyakit Jantung Bawaan (diagnosis yang tertulis
misalnya TOF, PDA, atresia trikuspid), bukan premature junctional beat. Ini mengubah arah klasifikasi (label per pasien
atau per lembar, bukan per denyut), jadi perlu disepakati dulu.

`tools/pair_pages.py` memasangkan tiap halaman EKG dengan formulir diagnosis pasien yang sama berdasarkan urutan halaman
di folder (separuh pertama formulir, separuh kedua EKG). Folder yang polanya tidak rapi ditandai untuk dicek manual.

## Langkah berikutnya yang disarankan

1. Cek acak 20 sampai 30 `overlay.jpg` dari halaman yang berbeda folder untuk memastikan kualitasnya sesuai harapan.
2. Isi label diagnosis di `pairs.csv` dari formulir (tulisan tangan, belum bisa otomatis).
3. Kalau ingin akurasi lebih tinggi pada strip yang sangat berisik, mask dari U-Net Open-ECG-Digitizer bisa dipasang
   sebagai pengganti threshold di `ecgdig/panel.py` (`panel_mask`); tahap lain tidak perlu diubah.
4. Halaman EDAN (folder 1-18) butuh layout tersendiri.
