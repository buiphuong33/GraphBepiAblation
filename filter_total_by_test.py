import pickle as pk
import pandas as pd

# ======================
# 1) Lấy danh sách ID từ test.pkl
# ======================
def load_test_ids(test_pkl_path):
    with open(test_pkl_path, "rb") as f:
        testset = pk.load(f)
    # mỗi sample có attribute .name
    test_ids = [sample.name.strip() for sample in testset]
    print(f"[INFO] Loaded {len(test_ids)} test IDs")
    return set(test_ids)  # dùng set cho nhanh


# ======================
# 2) Lọc total.csv theo test IDs
# ======================
def filter_total_csv(total_csv_path, test_ids, out_path="test_total.csv"):
    df = pd.read_csv(total_csv_path)

    # cột chứa ID trong total.csv là: "PDB chain"
    df["PDB chain"] = df["PDB chain"].astype(str).str.strip()

    df_test = df[df["PDB chain"].isin(test_ids)]
    df_test.to_csv(out_path, index=False)

    print(f"[DONE] Saved {len(df_test)} rows → {out_path}")


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    root = "./BCE_633"

    test_ids = load_test_ids(f"{root}/test.pkl")

    filter_total_csv(
        total_csv_path=f"{root}/total.csv",      # file bạn gửi format
        test_ids=test_ids,
        out_path=f"{root}/test_total.csv"
    )
