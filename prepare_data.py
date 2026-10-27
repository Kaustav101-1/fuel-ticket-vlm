"""
STEP 1 OF 3 — DATA PREPARATION
==============================
Turns the labelled Excel sheet + ticket images into the train / validation / test
files that the fine-tuning script (step 2) reads.

What it does, in order:
  1. Reads the ground-truth labels from the Excel sheet.
  2. Drops tickets that are unusable (badly faded / special cases).
  3. Cleans each label into a consistent JSON "answer" (the target the model learns).
  4. Optionally shrinks images so they upload and train faster (handwriting stays legible).
  5. Splits tickets into train / validation / test sets (fixed random seed = reproducible).
  6. Writes each set as JSONL in the chat format Hugging Face TRL expects for
     vision-language models: [image + instruction]  ->  [JSON answer].

Run:
    python prepare_data.py --excel fuel_tickets_labelled.xlsm --images images --out data

Folder layout expected:
    fuel_project/
      fuel_tickets_labelled.xlsx
      images/ticket_001.jpg ... ticket_240.jpg
      prepare_data.py
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — the knobs you may want to change
# ---------------------------------------------------------------------------

# The fields the model must extract, in the order they appear in the answer.
# (Excel column name -> JSON key the model will output.)
FIELDS = {
    "FDV_No": "fdv_no",
    "Date": "date",
    "IATA_Code": "iata_code",
    "SAP_Code": "sap_code",
    "Customer": "customer",
    "Flight_No": "flight_no",
    "Aircraft_Reg": "aircraft_reg",
    "Aircraft_Type": "aircraft_type",
    "Domestic / Export_Bonded": "domestic_or_export_bonded",
    "Proceeding_To": "proceeding_to",
    "MTOW_<40T": "mtow_below_40t",
    "Equip_Positioned": "equip_positioned",
    "Fuelling_Started": "fuelling_started",
    "Fuelling_Completed": "fuelling_completed",
    "Fuel_Grade": "fuel_grade",
    "Fuel_Batch_No": "fuel_batch_no",
    "Equipment_No": "equipment_no",
    "Opening_Meter": "opening_meter",
    "Closing_Meter": "closing_meter",
    "Difference_L": "difference_litres",
    "Total_Qty_L": "total_qty_litres",
    "Qty_In_Words": "qty_in_words",
    "Density_kg/m3": "density_kg_m3",
    "Temp_C": "temp_c",
}
INT_FIELDS = {"opening_meter", "closing_meter", "difference_litres", "total_qty_litres", "density_kg_m3"}
FLOAT_FIELDS = {"temp_c"}

# Tickets left out of the dataset entirely: unreadable scans or one-off special
# cases whose "correct answer" is not a clean single value.
EXCLUDE = {
    "ticket_016.jpg",  # severely blurred
    "ticket_204.jpg",  # severely faded
    "ticket_225.jpg",  # bowser swapped mid-fuelling, three meter sets
    "ticket_229.jpg",  # severely faded, quantity blank
}

# Split proportions. Test set is held back until the very end and is the only
# number you should quote to the panel.
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10        # test = remaining 0.20
SEED = 42                                # fixed seed -> same split every run

# Longest image side after resizing (pixels). Scans are ~1700x2200 at 200 DPI.
# 1600 keeps handwriting sharp while cutting file size roughly in half.
# Set to 0 to keep originals.
MAX_SIDE = 1600

# The instruction shown to the model with every image. The SAME text is used at
# inference time (step 3) — if you change it here, change it there too.
PROMPT = (
    "You are reading an Indian Oil Corporation aviation Fuel Delivery Voucher. "
    "Extract the following fields and reply with ONLY a JSON object using exactly these keys: "
    + ", ".join(FIELDS.values())
    + ". Rules: copy values as written on the voucher; dates as YYYY-MM-DD; times as HH:MM (24h); "
    "numbers without units or commas; use \"ILLEGIBLE\" if a value is present but unreadable "
    "and \"Not marked\" if a box or field is left blank."
)

# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------

def clean_value(key: str, raw: str):
    """Turn one Excel cell into a consistent JSON value.
    Numbers become real numbers (so 790 and 790.0 are the same target);
    the special labels ILLEGIBLE / Not marked stay as text."""
    raw = str(raw).strip()
    if raw in ("ILLEGIBLE", "Not marked", "NOT VISIBLE", ""):
        return raw or "Not marked"
    try:
        if key in INT_FIELDS:
            return int(round(float(raw)))
        if key in FLOAT_FIELDS:
            return round(float(raw), 1)
    except ValueError:
        pass  # e.g. "MULTIPLE" — keep as text
    return raw


def build_target(row) -> str:
    """The answer the model learns to produce: one compact JSON string."""
    obj = {key: clean_value(key, row[col]) for col, key in FIELDS.items()}
    return json.dumps(obj, ensure_ascii=False)


def prepare_image(src: Path, dst: Path) -> None:
    """Copy one image, converting to RGB and shrinking if larger than MAX_SIDE.
    Aspect ratio is preserved — no stretching, no cropping — because distorting
    handwriting hurts recognition. Padding/patching is handled later by the
    model's own processor."""
    img = Image.open(src).convert("RGB")
    if MAX_SIDE and max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    img.save(dst, "JPEG", quality=90)


def to_record(row, image_rel_path: str) -> dict:
    """One training example in the chat format TRL's SFTTrainer expects for VLMs.
    The user turn holds the image + instruction; the assistant turn holds the answer."""
    return {
        "id": row["Image_File"].removesuffix(".jpg"),
        "images": [image_rel_path],
        "messages": [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": build_target(row)},
            ]},
        ],
    }

# ---------------------------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="fuel_tickets_labelled.xlsm")  # .xlsx also works
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # Read every cell as text so nothing is silently reformatted by pandas.
    df = pd.read_excel(args.excel, sheet_name="Labelled Data", dtype=str, keep_default_na=False)
    df = df[list(["Image_File"] + list(FIELDS))]
    print(f"Labelled tickets in Excel : {len(df)}")

    # Drop excluded tickets and any row whose image is missing.
    n_before = len(df)
    df = df[~df["Image_File"].isin(EXCLUDE)]
    n_excluded = n_before - len(df)
    missing = [f for f in df["Image_File"] if not (Path(args.images) / f).exists()]
    if missing:
        print(f"WARNING: {len(missing)} images not found, skipped: {missing[:5]}...")
    df = df[~df["Image_File"].isin(missing)].reset_index(drop=True)
    print(f"Usable tickets            : {len(df)}  (excluded {n_excluded})")

    # Shuffle once with a fixed seed, then cut into train / val / test.
    idx = list(range(len(df)))
    random.Random(SEED).shuffle(idx)
    n_train = int(len(idx) * TRAIN_FRAC)
    n_val = int(len(idx) * VAL_FRAC)
    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }

    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i in sorted(rows):
                row = df.iloc[i]
                rel = f"images/{row['Image_File']}"
                prepare_image(Path(args.images) / row["Image_File"], out / rel)
                f.write(json.dumps(to_record(row, rel), ensure_ascii=False) + "\n")
        print(f"{name:5s}: {len(rows):3d} tickets -> {out / (name + '.jsonl')}")

    # Save the test-set ticket list: these labels must be hand-verified, because
    # the final accuracy you report is measured against them.
    test_ids = sorted(df.iloc[i]["Image_File"] for i in splits["test"])
    (out / "test_tickets_to_verify.txt").write_text("\n".join(test_ids) + "\n")
    print(f"Test tickets to hand-verify listed in {out / 'test_tickets_to_verify.txt'}")


if __name__ == "__main__":
    main()
