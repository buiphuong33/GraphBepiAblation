import os
import pickle as pk
import shutil
from tqdm import tqdm

def extract_test_pdb(root="./BCE_633", 
                     pdb_folder="purePDB",
                     output_folder="test_pdb_for_discotope"):
    """
    Đọc test.pkl → lấy danh sách sample test → copy các file .pdb tương ứng
    sang thư mục mới output_folder.
    """

    test_pkl = os.path.join(root, "test.pkl")

    if not os.path.exists(test_pkl):
        raise FileNotFoundError(f"Không tìm thấy file test.pkl tại: {test_pkl}")

    # Load test set
    with open(test_pkl, "rb") as f:
        testset = pk.load(f)

    # Folder chứa tất cả pdb (train + test)
    pdb_src = os.path.join(root, pdb_folder)

    # Folder output chứa pdb của test
    test_pdb_dir = os.path.join(root, output_folder)
    os.makedirs(test_pdb_dir, exist_ok=True)

    copied, missing = 0, []

    print(f"[INFO] Copy PDB files from {pdb_src} → {test_pdb_dir}")
    for it in tqdm(testset):
        pdb_name = f"{it.name}.pdb"       # GraphBepi sample → tên PDB
        src = os.path.join(pdb_src, pdb_name)
        dst = os.path.join(test_pdb_dir, pdb_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
        else:
            missing.append(pdb_name)

    print(f"[DONE] Copied {copied} PDB test files.")
    if missing:
        print(f"[WARN] {len(missing)} PDB missing in source folder:")
        print(missing[:10])

    return test_pdb_dir


if __name__ == "__main__":
    # Tùy chỉnh đường dẫn root nếu cần
    root = "./BCE_633"

    extract_test_pdb(
        root=root,
        pdb_folder="purePDB",                # đổi thành 'PDB' nếu bạn dùng folder này
        output_folder="test_pdb_for_discotope"
    )
