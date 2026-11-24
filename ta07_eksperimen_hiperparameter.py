# %% [markdown]
# ## Langkah 5: Eksperimen Hiperparameter
#
# Pilih **satu** hiperparameter saja (`gamma` ATAU `alpha`), lalu latih model dengan nilai default (tinggi)
# dan nilai ekstrem (sangat rendah). Untuk `gamma`, nilai menengah 0,1 ikut diuji sebagai pembanding.
#
# - **gamma rendah:** agen menjadi *miopik*, hanya peduli reward instan. Reward +20 saat antar paket
#   tidak "terasa" dari state yang jauh, sehingga agen sulit belajar rute panjang.
# - **alpha rendah:** pembelajaran sangat lambat karena nilai Q hanya bergeser sedikit tiap update.
#   (alpha terlalu tinggi bisa membuat Q-Table tidak stabil.)

# %% [markdown]
# **Catatan:** pada environment deterministik ini, gamma = 0.1 ternyata masih berhasil, karena selisih
# nilai Q yang kecil tetap terbaca oleh `argmax`. Efek "miopik" baru jelas pada nilai yang jauh lebih
# rendah (0.01 atau 0.0), sehingga nilai ekstrem yang dipakai adalah 0.01.

# %%
from ta07_taxi_qlearning import *  # [NB-SKIP]

PARAM = "gamma"   # ganti ke "alpha" untuk eksperimen learning rate
EKSPERIMEN = {
    # urutan: default (tinggi) -> menengah (opsional) -> ekstrem (sangat rendah)
    "gamma": {"default": 0.9, "menengah": 0.1, "ekstrem": 0.01},
    "alpha": {"default": 0.8, "ekstrem": 0.01},
}


def jalankan_eksperimen(param):
    runs, ringkasan, detail = {}, [], []
    for jenis, nilai in EKSPERIMEN[param].items():
        hp = dict(HP)
        hp[param] = nilai
        label = f"{param}={nilai} ({jenis})"
        print(f"\n=== Pelatihan: {label} ===")
        env = TaxiEnv(seed=SEED)
        res = train_q_learning(env, seed=SEED, **hp)
        runs[label] = res

        ma = moving_average(res["rewards"], 100)
        ep_konv, _ = find_convergence(ma)
        ev = evaluate_policy(res["q"])
        ra = route_analysis(res["q"])
        _, info = training_summary(res)
        t, m, k = ra["total"], ra["menunggu"], ra["kendaraan"]
        ringkasan.append({
            "Model": label,
            "Reward 100 ep. terakhir": round(float(np.mean(res["rewards"][-100:])), 2),
            "Konvergensi (episode)": ep_konv,
            "Sukses greedy (%)": round(ev["success_rate"] * 100, 1),
            "Reward greedy": round(ev["avg_reward"], 2),
            "Langkah rata-rata": round(ev["avg_steps"], 1),
        })
        detail.append({
            "Model": label,
            "State awal berhasil": f"{t['sukses']}/{t['n']}",
            "Rute terpendek": f"{t['terpendek']}/{t['n']}",
            "Paket menunggu (berhasil)": f"{m['sukses']}/{m['n']}",
            "Paket di kendaraan (berhasil)": f"{k['sukses']}/{k['n']}",
            "Sukses saat pelatihan, per 100 ep. (ep. 1000-5000)":
                " / ".join(f"{v * 100:.0f}%" for v in info["sukses_pelatihan_per_100"].values()),
        })
    return runs, pd.DataFrame(ringkasan), pd.DataFrame(detail)


# %%
if __name__ == "__main__":  # [NB-RUN]
    runs, tabel, detail = jalankan_eksperimen(PARAM)

    # %% [markdown]
    # ### Perbandingan kurva pembelajaran

    # %%
    plot_learning_curves(runs, window=100,
                         title=f"Perbandingan Kurva Pembelajaran ({PARAM})",
                         save_path=f"output/eksperimen_{PARAM}_kurva.png")

    # %% [markdown]
    # ### Perbandingan kebijakan pada fase mengantar paket (paket di kendaraan, tujuan G)
    # (model default dibandingkan dengan model ekstrem)

    # %%
    labels = list(runs.keys())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, label in zip(axes, [labels[0], labels[-1]]):
        plot_policy(runs[label]["q"], pass_loc=4, dest=1, ax=ax, title=label)
    fig.suptitle("Kebijakan fase mengantar paket (· = belum ada preferensi aksi)")
    fig.savefig(f"output/eksperimen_{PARAM}_policy.png", dpi=150, bbox_inches="tight")
    plt.show()

    # %% [markdown]
    # ### Perbandingan kebijakan pada fase menuju paket (paket di R, tujuan G)

    # %%
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, label in zip(axes, [labels[0], labels[-1]]):
        plot_policy(runs[label]["q"], pass_loc=0, dest=1, ax=ax, title=label)
    fig.suptitle("Kebijakan fase menuju paket")
    fig.savefig(f"output/eksperimen_{PARAM}_policy_fase1.png", dpi=150, bbox_inches="tight")
    plt.show()

    # %% [markdown]
    # ### Tabel ringkasan
    #
    # Gunakan tabel ini sebagai bahan analisis di laporan (Langkah 5).
    # Tabel kedua merinci keberhasilan kebijakan dari seluruh 400 state awal yang mungkin.

    # %%
    print(tabel.to_string(index=False))
    print()
    print(detail.to_string(index=False))
    tabel.to_csv(f"output/eksperimen_{PARAM}_ringkasan.csv", index=False)
    detail.to_csv(f"output/eksperimen_{PARAM}_rincian.csv", index=False)
