import io, os, sys, contextlib
import numpy as np
import pandas as pd
from ta07_taxi_qlearning import (TaxiEnv, HP, SEED, train_q_learning, moving_average, find_convergence,
                                 evaluate_policy, demo_episode, training_summary, route_analysis, q_table_terisi)

hasil = []   # (kelompok, klaim, nilai di laporan, nilai dari kode, lulus)


def cek(kelompok, klaim, laporan, kode, tol=0.0):
    ok = abs(float(laporan) - float(kode)) <= tol + 1e-9
    hasil.append((kelompok, klaim, laporan, round(float(kode), 4), ok))


def latih(gamma):
    hp = dict(HP); hp["gamma"] = gamma
    with contextlib.redirect_stdout(io.StringIO()):
        return train_q_learning(TaxiEnv(seed=SEED), seed=SEED, **hp)


# ---------------------------------------------------------------- pelatihan utama (seed 42, gamma 0.9)
A = latih(0.9)
r = A["rewards"]; ma = moving_average(r, 100)
conv, akhir = find_convergence(ma)
G = "Pelatihan utama"
cek(G, "Reward 100 episode pertama", -690.12, np.mean(r[:100]), 0.005)
cek(G, "Reward 100 episode terakhir", 8.72, np.mean(r[-100:]), 0.005)
cek(G, "Episode konvergensi", 3474, conv)
cek(G, "Nilai akhir MA saat konvergensi", 8.97, akhir, 0.005)
tabel, info = training_summary(A)
cek(G, "Pertama kali MA-100 > 0 (episode)", 1822, info["pertama_melewati_nol"])
cek(G, "Simpangan baku reward, 500 pertama", 275.9, info["std_500_pertama"], 0.05)
cek(G, "Simpangan baku reward, 500 terakhir", 3.9, info["std_500_terakhir"], 0.05)
cek(G, "Episode berhasil, episode 1-100 (%)", 22, info["sukses_100_pertama"] * 100, 0.5)
cek(G, "Episode berhasil, episode 151-250 (%)", 93, info["sukses_episode_151_250"] * 100, 0.5)
klaim_tabel = {100: (-690.12, 0.907), 250: (-275.93, 0.782), 500: (-85.78, 0.611), 1000: (-22.96, 0.375),
               2000: (0.38, 0.144), 3000: (7.92, 0.059), 3474: (6.95, 0.041), 5000: (8.72, 0.017)}
for _, row in tabel.iterrows():
    ep = int(row["episode"]); m_, e_ = klaim_tabel[ep]
    cek("Tabel titik pelatihan", f"MA-100 episode {ep}", m_, row["ma100"], 0.005)
    cek("Tabel titik pelatihan", f"epsilon episode {ep}", e_, row["epsilon"], 0.0005)

ev = evaluate_policy(A["q"])
cek(G, "Evaluasi greedy: keberhasilan (%)", 100.0, ev["success_rate"] * 100)
cek(G, "Evaluasi greedy: rata-rata reward", 9.35, ev["avg_reward"], 0.005)
cek(G, "Evaluasi greedy: rata-rata langkah", 11.7, ev["avg_steps"], 0.05)
cek(G, "State dengan nilai Q terisi", 400, q_table_terisi(A["q"]))

ra = route_analysis(A["q"])
cek(G, "State awal berhasil (dari 400)", 400, ra["total"]["sukses"])
cek(G, "Rute terpendek (dari 400)", 394, ra["total"]["terpendek"])
cek(G, "Rute terpendek, paket menunggu (dari 300)", 294, ra["menunggu"]["terpendek"])
cek(G, "Rute terpendek, paket di kendaraan (dari 100)", 100, ra["kendaraan"]["terpendek"])
lp = ra["lebih_panjang"]
cek(G, "Jumlah state dengan rute lebih panjang", 6, len(lp))
cek(G, "Selisih langkah maksimum pada state tsb", 2, (lp["langkah"] - lp["minimum"]).max())
cek(G, "Seluruhnya pada fase menuju paket (paket < 4)", 1, int((lp["paket"] < 4).all()))
c = lp[(lp.baris == 2) & (lp.kolom == 4) & (lp.paket == 0) & (lp.tujuan == 1)].iloc[0]
cek(G, "Contoh (2,4) paket R tujuan G: langkah", 14, c["langkah"])
cek(G, "Contoh (2,4) paket R tujuan G: minimum", 12, c["minimum"])

jejak = demo_episode(A["q"])
cek(G, "Contoh episode: jumlah langkah", 9, len(jejak))
cek(G, "Contoh episode: total reward", 12, jejak["reward"].sum())

# ---------------------------------------------------------------- eksperimen gamma (seed 42)
klaim_g = {0.9: dict(r100=8.72, konv=3474, sukses=100.0, rgreedy=9.35, langkah=11.7, menunggu=300, kendaraan=100),
           0.1: dict(r100=8.50, konv=3624, sukses=100.0, rgreedy=9.38, langkah=11.6, menunggu=300, kendaraan=100),
           0.01: dict(r100=-38.48, konv=5000, sukses=37.5, rgreedy=-120.52, langkah=9.0, menunggu=120, kendaraan=100)}
for g, kl in klaim_g.items():
    res = A if g == 0.9 else latih(g)
    m_ = moving_average(res["rewards"], 100); cv, _ = find_convergence(m_)
    e_ = evaluate_policy(res["q"]); rr = route_analysis(res["q"])
    K = f"Eksperimen gamma={g} (seed 42)"
    cek(K, "Reward 100 episode terakhir", kl["r100"], np.mean(res["rewards"][-100:]), 0.005)
    cek(K, "Konvergensi (episode)", kl["konv"], cv)
    cek(K, "Sukses greedy (%)", kl["sukses"], e_["success_rate"] * 100, 0.05)
    cek(K, "Reward greedy", kl["rgreedy"], e_["avg_reward"], 0.005)
    cek(K, "Langkah rata-rata (episode berhasil)", kl["langkah"], e_["avg_steps"], 0.05)
    cek(K, "State awal berhasil, paket menunggu", kl["menunggu"], rr["menunggu"]["sukses"])
    cek(K, "State awal berhasil, paket di kendaraan", kl["kendaraan"], rr["kendaraan"]["sukses"])
    if g == 0.01:
        _, inf = training_summary(res)
        v = [x * 100 for x in inf["sukses_pelatihan_per_100"].values()]
        cek(K, "Sukses saat pelatihan, minimum (%)", 83, min(v), 0.5)
        cek(K, "Sukses saat pelatihan, maksimum (%)", 91, max(v), 0.5)

# ---------------------------------------------------------------- 10 seed (Tabel 7 makalah jurnal)
csv = next((p for p in ("output/multi_seed_hasil.csv", "ta07_multi_seed_hasil.csv") if os.path.exists(p)), None)
if csv is None:
    print("output/multi_seed_hasil.csv belum ada. Jalankan ta07_multi_seed.py dulu.")
else:
    df = pd.read_csv(csv)
    klaim10 = {0.9: dict(r100=(8.74, 0.71), konv=(3362, 238), sg=100.0, ss=100.0, rt=(99.4, 0.4)),
               0.1: dict(r100=(8.79, 0.54), konv=(3386, 132), sg=100.0, ss=99.98, rt=(99.98, 0.08)),
               0.01: dict(r100=(-39.99, 10.64), konv=(4994, 8), sg=37.5, ss=55.0, rt=(55.0, 0.0))}
    for g, kl in klaim10.items():
        d = df[np.isclose(df["gamma"], g)]
        K = f"10 seed, gamma={g}"
        cek(K, "Jumlah seed", 10, len(d))
        cek(K, "Reward 100 ep. akhir (rata-rata)", kl["r100"][0], d["reward_100"].mean(), 0.005)
        cek(K, "Reward 100 ep. akhir (simpangan baku)", kl["r100"][1], d["reward_100"].std(), 0.005)
        cek(K, "Konvergensi (rata-rata)", kl["konv"][0], d["konvergensi"].mean(), 0.5)
        cek(K, "Konvergensi (simpangan baku)", kl["konv"][1], d["konvergensi"].std(), 0.5)
        cek(K, "Sukses greedy (%)", kl["sg"], d["sukses_greedy"].mean() * 100, 0.05)
        cek(K, "State awal berhasil (%)", kl["ss"], d["semua_state_sukses"].mean() * 100, 0.005)
        cek(K, "Rute terpendek (rata-rata %)", kl["rt"][0], d["rute_terpendek"].mean() * 100, 0.05 if g == 0.9 else 0.005)
        cek(K, "Rute terpendek (simpangan baku, poin %)", kl["rt"][1], d["rute_terpendek"].std() * 100, 0.05 if g == 0.9 else 0.005)
        if g == 0.1:
            cek(K, "Jumlah state awal yang gagal (dari 4.000)", 1, round((1 - d["semua_state_sukses"]).sum() * 400))
            cek(K, "Seed yang mengalami kegagalan tsb", 7, int(d.loc[d["semua_state_sukses"] < 1, "seed"].iloc[0]))
        if g == 0.01:
            cek(K, "Sukses greedy identik di semua seed (std)", 0.0, d["sukses_greedy"].std(), 1e-9)
            cek(K, "State awal berhasil identik di semua seed (std)", 0.0, d["semua_state_sukses"].std(), 1e-9)

# ---------------------------------------------------------------- laporan
df_h = pd.DataFrame(hasil, columns=["Kelompok", "Klaim", "Di laporan", "Dari kode", "Cocok"])
df_h["Cocok"] = df_h["Cocok"].map({True: "ya", False: "TIDAK"})
pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 60)
print(df_h.to_string(index=False))
gagal = (df_h["Cocok"] == "TIDAK").sum()
print(f"\nTotal klaim diperiksa: {len(df_h)} | cocok: {len(df_h) - gagal} | tidak cocok: {gagal}")
sys.exit(1 if gagal else 0)
