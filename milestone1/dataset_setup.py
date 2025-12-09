import os
import cv2
import pandas as pd

# ================== EDIT THESE TWO PATHS ==================
# Full path to your TraceFinder_AI_Dataset folder
RAW_BASE = r"C:\Techie\Projects\TraceFinder_AI_Dataset"

# Where you want processed images to be saved
PROC_BASE = r"C:\Techie\Projects\TraceFinder_Processed"
# ==========================================================

LABEL_CSV = os.path.join(os.path.dirname(__file__), "tracefinder_labels.csv")
IMG_SIZE = 256
SOURCES = ["Flatfield", "Official", "Wikipedia"]


def build_labels():
    """Scan the dataset folders and build a CSV with image paths + labels."""
    rows = []

    for src in SOURCES:
        src_dir = os.path.join(RAW_BASE, src)
        if not os.path.isdir(src_dir):
            print(f"[WARN] Source folder not found: {src_dir}")
            continue

        for scanner in os.listdir(src_dir):
            scanner_dir = os.path.join(src_dir, scanner)
            if not os.path.isdir(scanner_dir):
                continue

            if src == "Flatfield":
                # Flatfield / scanner / *.tif
                for fname in os.listdir(scanner_dir):
                    if fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                        rel_path = os.path.join(src, scanner, fname)
                        rows.append({
                            "rel_path": rel_path,
                            "source_type": src,
                            "scanner": scanner,
                            "dpi": None,
                            "doc_id": os.path.splitext(fname)[0],
                        })
            else:
                # Official or Wikipedia: Source / scanner / dpi / *.tif
                for dpi in os.listdir(scanner_dir):
                    dpi_dir = os.path.join(scanner_dir, dpi)
                    if not os.path.isdir(dpi_dir):
                        continue
                    for fname in os.listdir(dpi_dir):
                        if fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                            rel_path = os.path.join(src, scanner, dpi, fname)
                            rows.append({
                                "rel_path": rel_path,
                                "source_type": src,
                                "scanner": scanner,
                                "dpi": dpi,
                                "doc_id": os.path.splitext(fname)[0],
                            })

    df = pd.DataFrame(rows)
    df.to_csv(LABEL_CSV, index=False)
    print(f"[OK] Saved labels to {LABEL_CSV} with {len(df)} images.")


def preprocess_images():
    """Convert all images to grayscale + resize, and save them by scanner."""
    os.makedirs(PROC_BASE, exist_ok=True)

    for src in SOURCES:
        src_dir = os.path.join(RAW_BASE, src)
        if not os.path.isdir(src_dir):
            print(f"[WARN] Source folder not found: {src_dir}")
            continue

        for scanner in os.listdir(src_dir):
            scanner_raw_dir = os.path.join(src_dir, scanner)
            if not os.path.isdir(scanner_raw_dir):
                continue

            # One processed folder per scanner
            scanner_proc_dir = os.path.join(PROC_BASE, scanner)
            os.makedirs(scanner_proc_dir, exist_ok=True)

            if src == "Flatfield":
                for fname in os.listdir(scanner_raw_dir):
                    if not fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                        continue
                    in_path = os.path.join(scanner_raw_dir, fname)

                    img = cv2.imread(in_path)
                    if img is None:
                        print(f"[WARN] Could not read {in_path}")
                        continue

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

                    out_name = f"{src}_NA_{fname}"
                    out_path = os.path.join(scanner_proc_dir, out_name)
                    cv2.imwrite(out_path, resized)
            else:
                for dpi in os.listdir(scanner_raw_dir):
                    dpi_dir = os.path.join(scanner_raw_dir, dpi)
                    if not os.path.isdir(dpi_dir):
                        continue
                    for fname in os.listdir(dpi_dir):
                        if not fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                            continue
                        in_path = os.path.join(dpi_dir, fname)

                        img = cv2.imread(in_path)
                        if img is None:
                            print(f"[WARN] Could not read {in_path}")
                            continue

                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

                        out_name = f"{src}_{dpi}_{fname}"
                        out_path = os.path.join(scanner_proc_dir, out_name)
                        cv2.imwrite(out_path, resized)

    print(f"[OK] Preprocessing finished. Images saved in {PROC_BASE}")


if __name__ == "__main__":
    print("=== Milestone 1: Dataset Setup ===")
    print("Step 1: Building labels CSV...")
    build_labels()
    print("Step 2: Preprocessing images...")
    preprocess_images()
    print("=== Done: Milestone 1 dataset ready ===")
