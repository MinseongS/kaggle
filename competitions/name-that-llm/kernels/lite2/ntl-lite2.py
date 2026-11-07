# ntl-lite2: blend of (a) TF-IDF LinearSVC on train + Open Pool sample and
# (b) frozen Qwen3-1.7B-Base mean-pooled layers 2,4,6,20 + LR head (fit locally on train + pool), + Sinkhorn.
# Local grouped CV: (a) 0.4708, (b) 0.4804, blend 1:1 0.5356 (+sinkhorn 0.5382).
import glob, json, time, numpy as np, pandas as pd, torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

T0 = time.time()
LABELS = ["claude-fable-5","claude-opus-4-6","claude-opus-4-8","claude-opus-5","claude-sonnet-5",
 "gemini-3.1-pro-preview","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash","glm-5.3","glm-5.3-flash",
 "gpt-5.5","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra","gpt-6-astra","kimi-k3","qwen3.8-max"]
L2I = {m: i for i, m in enumerate(LABELS)}
def find(pat): return glob.glob(f"/kaggle/input/**/{pat}", recursive=True)[0]
comp = find("name-that-llm/train.csv").rsplit("/", 1)[0]
tr = pd.read_csv(f"{comp}/train.csv"); te = pd.read_csv(f"{comp}/test.csv")
y_tr = tr.model.map(L2I).values

# ---------- (b) embedding member ----------
head = np.load(find("emb_head.npz")); meta = json.load(open(find("emb_head.json")))
assert meta["labels"] == LABELS
mdir = find("config.json").rsplit("/", 1)[0]
print("model dir", mdir, flush=True)
tok = AutoTokenizer.from_pretrained(mdir); tok.padding_side = "right"
lm = AutoModel.from_pretrained(mdir, dtype=torch.float16).cuda().eval()
L = meta["hidden_layers"]; lm.layers = lm.layers[: max(L) + 1]  # keep one extra so hidden_states[max(L)] is pre-norm

@torch.no_grad()
def embed(texts, bs=32):
    order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), len(L) * lm.config.hidden_size), np.float32)
    for s in range(0, len(texts), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=meta["max_len"]).to("cuda")
        hs = lm(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float(); m[:, 0] = 0
        out[idx] = torch.cat([(hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1) for l in L], 1).cpu().numpy()
    return out

def emb_prob(X):
    z = ((X - head["mean"]) / head["scale"]) @ head["coef"].T + head["intercept"]
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

chk = emb_prob(embed(tr.text.tolist()[:600]))
print(f"emb head in-sample acc on 600 train rows: {(chk.argmax(1) == y_tr[:600]).mean():.3f}  ({time.time()-T0:.0f}s)", flush=True)
PB = emb_prob(embed(te.text.tolist()))
print(f"emb test done {time.time()-T0:.0f}s", flush=True)
del lm; torch.cuda.empty_cache()

# ---------- (a) TF-IDF member ----------
p = pd.read_parquet(find("ntl_open_pool.parquet"))
p = p[~(p.flag_near_train_prompt | p.flag_train_prompt_regen | p.flag_gemini37_routing_risk)]
rng = np.random.default_rng(0)
p = p.assign(_k=np.log(rng.random(len(p))) / p.match_weight.clip(lower=1e-6).values)
p = p.sort_values("_k", ascending=False).groupby("model").head(2500)
texts = tr.text.tolist() + p.text.tolist()
y = np.concatenate([y_tr, p.model.map(L2I).values])
w = np.concatenate([np.ones(len(tr)), np.full(len(p), 0.5)])
vs = [TfidfVectorizer(analyzer="char", ngram_range=(1, 5), sublinear_tf=True, lowercase=False, min_df=3,
                      max_features=400_000, dtype=np.float32),
      TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, lowercase=False, min_df=2,
                      max_features=200_000, token_pattern=r"(?u)\b\w+\b|[^\w\s]", dtype=np.float32)]
Xa = hstack([v.fit_transform(texts) for v in vs]).tocsr()
Xt = hstack([v.transform(te.text) for v in vs]).tocsr()
d = LinearSVC(C=0.3, class_weight="balanced").fit(Xa, y, sample_weight=w).decision_function(Xt) / 0.1
PA = np.exp(d - d.max(1, keepdims=True)); PA /= PA.sum(1, keepdims=True)
print(f"tfidf done {time.time()-T0:.0f}s", flush=True)

# ---------- blend + sinkhorn ----------
lp = 0.5 * np.log(PA + 1e-9) + 0.5 * np.log(PB + 1e-9)
P = np.exp(lp - lp.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
np.save("probs_tfidf.npy", PA); np.save("probs_emb.npy", PB)
if len(te) >= 18:
    for _ in range(50):
        P = P / P.sum(0, keepdims=True) * (len(P) / 18); P = P / P.sum(1, keepdims=True)
sub = pd.DataFrame({"id": te.id, "model": [LABELS[i] for i in P.argmax(1)]})
sub.to_parquet("submission.parquet", index=False)
print(sub.model.value_counts().to_string(), f"\ntotal {time.time()-T0:.0f}s")
