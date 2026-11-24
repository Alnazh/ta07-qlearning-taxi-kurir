"""Uji kekokohan: ulangi pelatihan dengan beberapa seed."""
import io, contextlib
import numpy as np
import pandas as pd
from ta07_taxi_qlearning import (TaxiEnv, HP, train_q_learning, moving_average,
                                 find_convergence, evaluate_policy)

SEEDS = list(range(10))
GAMMAS = [0.9, 0.1, 0.01]
P = TaxiEnv.PICKUPS


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def route_stats(q):
    """Bandingkan panjang rute greedy dengan jarak minimum dari 400 state awal."""
    ok = opt = 0
    for row in range(5):
        for col in range(5):
            for p in range(5):
                for d in range(4):
                    if p == d:
                        continue
                    env = TaxiEnv(seed=0)
                    s = env.state = TaxiEnv.encode(row, col, p, d)
                    n, done = 0, False
                    while n < 200 and not done:
                        s, _, done = env.step(int(np.argmax(q[s])))
                        n += 1
                    best = (manhattan((row, col), P[p]) + 1 + manhattan(P[p], P[d]) + 1
                            if p < 4 else manhattan((row, col), P[d]) + 1)
                    ok += done
                    opt += (done and n == best)
    return ok / 400, opt / 400


rows = []
for g in GAMMAS:
    for sd in SEEDS:
        hp = dict(HP)
        hp["gamma"] = g
        with contextlib.redirect_stdout(io.StringIO()):
            res = train_q_learning(TaxiEnv(seed=sd), seed=sd, **hp)
        ma = moving_average(res["rewards"], 100)
        conv, _ = find_convergence(ma)
        ev = evaluate_policy(res["q"])
        ok, opt = route_stats(res["q"])
        rows.append(dict(gamma=g, seed=sd, reward_100=float(np.mean(res["rewards"][-100:])),
                         konvergensi=conv, sukses_greedy=ev["success_rate"],
                         reward_greedy=ev["avg_reward"], semua_state_sukses=ok, rute_terpendek=opt))
df = pd.DataFrame(rows)
df.to_csv("output/multi_seed_hasil.csv", index=False)
agg = df.groupby("gamma").agg(["mean", "std"]).round(3)
print(agg.to_string())
print(df.groupby("gamma")[["konvergensi"]].agg(["min", "max"]))
