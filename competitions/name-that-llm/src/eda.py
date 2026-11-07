import pandas as pd, numpy as np, re
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
tr = pd.read_csv("data/train.csv")
print(tr.shape, tr.prompt_id.nunique())
print(tr.model.value_counts().to_string())
print("passages/prompt:", tr.groupby("prompt_id").size().describe().round(1).to_dict())
print("models/prompt:", tr.groupby("prompt_id").model.nunique().describe().round(1).to_dict())
print("dup (prompt,model):", tr.duplicated(["prompt_id","model"]).sum())
tr["words"] = tr.text.str.split().str.len()
feats = {
 "words": tr.words,
 "emdash": tr.text.str.count("—")/tr.words*100,
 "endash": tr.text.str.count("–")/tr.words*100,
 "curly_apos": tr.text.str.contains("[’‘]").astype(int),
 "curly_dq": tr.text.str.contains("[“”]").astype(int),
 "straight_apos": tr.text.str.contains("'").astype(int),
 "newline": tr.text.str.contains("\n").astype(int),
 "md_bold": tr.text.str.contains(r"\*\*").astype(int),
 "md_ital": tr.text.str.contains(r"(?<!\*)\*[^*\s]").astype(int),
 "latex": tr.text.str.contains(r"\$|\\\(").astype(int),
 "semicolon": tr.text.str.count(";")/tr.words*100,
 "colon": tr.text.str.count(":")/tr.words*100,
 "nonascii": tr.text.apply(lambda s: sum(ord(c)>127 for c in s))/tr.words*100,
 "avg_wlen": tr.text.apply(lambda s: np.mean([len(w) for w in s.split()])),
 "sent_len": tr.text.apply(lambda s: len(s.split())/max(1,len(re.findall(r"[.!?](\s|$)", s)))),
}
F = pd.DataFrame(feats); F["model"] = tr.model
print(F.groupby("model").mean().round(2).to_string())
print("\nwords by model quantiles"); print(tr.groupby("model").words.quantile([.05,.5,.95]).unstack().round(0).to_string())
# first-word
tr["first"] = tr.text.str.split().str[0]
print("\ntop first words:"); print(tr["first"].value_counts().head(15).to_string())
