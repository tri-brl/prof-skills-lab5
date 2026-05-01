import csv
import re
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import soundfile as sf
import pandas as pd
from itertools import combinations

warnings.filterwarnings("ignore")


# ── config ────────────────────────────────────────────────────────────────────
CORPUS_ROOT  = Path(r"C:\Users\aviba\Downloads\ru-fr_interference\ru-fr_interference")
WAV_DIR      = CORPUS_ROOT / "2" / "wav_et_textgrids" / "FRcorp_textgrids_only" 
RUFR_CSV     = CORPUS_ROOT / "2"/ "RUFRcorr.csv"
META_CSV     = CORPUS_ROOT / "2"/ "metadata_RUFR.csv"
OUTPUT_DIR   = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME   = "facebook/wav2vec2-base"
TARGET_SR    = 16_000
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Set to True to process only the first 2 speakers (fast sanity check)
DRY_RUN      = False

# ── 1. parse RUFR.csv — build word -> [occ1, occ2, ...] index ─────────────────
def load_word_list(path: Path) -> dict[str, list[int]]:
    """
    Returns e.g. {'tsarine': [13,15,32,50,57,78], 'sérieux': [2,23,31,...], ...}
    The occurrence numbers are 1-based positions in the read-aloud list.
    """
    word_occurrences = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header (Word / Ipa / occ.1 / ...)
        for row in reader:
            if not row or not row[0].strip():
                continue
            word = row[0].strip()
            occs = [int(x.strip()) for x in row[2:] if x.strip().isdigit()]
            word_occurrences[word] = occs
    print(f"  Loaded {len(word_occurrences)} words from RUFR.csv")
    return word_occurrences

# word_occurrences = load_word_list(RUFR_CSV)

# for word, occs in word_occurrences.items():
#     print(word, "->", occs)


# ── 2. parse metadata.csv — speaker info ──────────────────────────────────────
def load_speaker_meta(path: Path) -> dict[str, dict]:
    speakers = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            spk_id = row[1].strip()
            speakers[spk_id] = {
                "L1":       row[2].strip(),
                "age":      row[3].strip(),
                "gender":   row[4].strip(),
                "FR_level": row[5].strip(),
                "RU_level": row[6].strip(),
            }
    print(f"  Loaded {len(speakers)} speakers from metadata.csv")
    return speakers

# speakers = load_speaker_meta(META_CSV)
# print(speakers)

# ── 3. parse a _words.csv — returns list of (word, start, end) ───────────────
def load_words_csv(path: Path) -> list[tuple[str, float, float]]:
    segments = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) != 3:
                continue
            word = row[0].strip().strip('"')
            try:
                start = float(row[1])
                end   = float(row[2])
            except ValueError:
                continue

            if word:  # skips "" (silence)
                segments.append((word, start, end))

    return segments

# segemnts = load_words_csv(WAV_DIR / "SD" / "sd_fra_list1_FRcorp42_words.csv")
# print(segemnts)


# ── 4. match segments to RUFR words by position ───────────────────────────────
def match_occurrences(
    segments: list[tuple[str, float, float]],
    word_occurrences: dict[str, list[int]],
) -> list[dict]:
    """
    The occurrence numbers in RUFR.csv are 1-based positions in the word list
    (ignoring silences). We enumerate the non-silent segments and check if the
    position matches any word's occurrence list.
    """
    matched = []
    # Build reverse map: position -> (word, repetition_index)
    pos_to_word = {}
    for word, occs in word_occurrences.items():
        for rep_idx, pos in enumerate(occs, start=1):
            pos_to_word[pos] = (word, rep_idx)

    for position, (seg_word, start, end) in enumerate(segments, start=1):
        if position in pos_to_word:
            word, rep = pos_to_word[position]
            matched.append({
                "word":       word,
                "repetition": rep,
                "start":      start,
                "end":        end,
                "seg_word":   seg_word,  # what the aligner labelled it
            })
    return matched


# ── 5. load model ─────────────────────────────────────────────────────────────
def load_model():
    print(f"\nLoading {MODEL_NAME} on {DEVICE} ...")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model     = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval().to(DEVICE)
    print(f"  Hidden size: {model.config.hidden_size}")
    return extractor, model


# # ── 6. extract one word slice ─────────────────────────────────────────────────
def extract_rep(
    waveform: torch.Tensor,
    sr: int,
    start: float,
    end: float,
    extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
) -> np.ndarray:

    # 1. resample full waveform first
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR

    # 2. slice AFTER resampling
    start_sample = int(start * sr)
    end_sample   = int(end * sr)

    slice_wav = waveform[:, start_sample:end_sample]

    if slice_wav.shape[1] == 0:
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    audio_np = slice_wav.squeeze().numpy().astype(np.float32)

    inputs = extractor(
        audio_np,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(inputs.input_values.to(DEVICE))

    hidden = outputs.last_hidden_state   # safer than hidden_states
    pooled = hidden.mean(dim=1).squeeze()

    return pooled.cpu().numpy().astype(np.float32)


# ── 7. main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading corpus resources ...")

    word_occurrences = load_word_list(RUFR_CSV)
    speaker_meta     = load_speaker_meta(META_CSV)
    extractor, model = load_model()

    speaker_dirs = sorted([d for d in WAV_DIR.iterdir() if d.is_dir()])

    if DRY_RUN:
        speaker_dirs = speaker_dirs[:2]
        print("\n[DRY RUN] processing first 2 speakers only")

    all_records = []
    representations = {}

    for spk_dir in speaker_dirs:
        spk_id = spk_dir.name.upper()
        print(f"\n── Speaker {spk_id} ──")

        wav_files  = sorted(spk_dir.glob("*.wav"))
        words_csvs = sorted(spk_dir.glob("*_words.csv"))

        if not wav_files:
            print("  [SKIP] no wav files found")
            continue

        wav_map = {w.stem: w for w in wav_files}
        pairs = []

        for wcsv in words_csvs:
            stem = re.sub(r"_words$", "", wcsv.stem)
            if stem in wav_map:
                pairs.append((wav_map[stem], wcsv))

        if not pairs and len(wav_files) == len(words_csvs):
            pairs = list(zip(sorted(wav_files), words_csvs))

        if not pairs:
            print("  [SKIP] could not pair wav + words.csv")
            continue

        for wav_path, words_csv_path in pairs:
            print(f"  Loading {wav_path.name} ...")

            # ── SAFE AUDIO LOADING (NO TORCHAUDIO) ─────────────
            audio, sr = sf.read(wav_path)

            waveform = torch.tensor(audio).float()

            # mono conversion if needed
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.mean(dim=1).unsqueeze(0)
            # ───────────────────────────────────────────────────

            segments = load_words_csv(words_csv_path)
            matched = match_occurrences(segments, word_occurrences)

            print(f"  Found {len(matched)} target word occurrences")

            for m in matched:
                key = f"{spk_id}__{m['word']}__{m['repetition']}"

                t0 = time.perf_counter()
                try:
                    rep = extract_rep(
                        waveform, sr,
                        m["start"], m["end"],
                        extractor, model
                    )

                    representations[key] = rep
                    elapsed = round(time.perf_counter() - t0, 3)

                    all_records.append({
                        "key": key,
                        "speaker": spk_id,
                        "word": m["word"],
                        "repetition": m["repetition"],
                        "start": round(m["start"], 4),
                        "end": round(m["end"], 4),
                        "wav": wav_path.name,
                        "elapsed": elapsed,
                        **speaker_meta.get(spk_id, {}),
                    })

                    print(f"    [{m['word']} rep{m['repetition']}] "
                          f"{m['start']:.2f}-{m['end']:.2f}s "
                          f"shape={rep.shape} {elapsed:.2f}s")

                except Exception as e:
                    print(f"    [ERROR] {key}: {e}")

    # ── SAVE EMBEDDINGS ─────────────────────────────────────
    npz_path = OUTPUT_DIR / "representations_float32.npz"
    np.savez_compressed(npz_path, **representations)

    size_mb = npz_path.stat().st_size / 1e6
    print(f"\nSaved {len(representations)} vectors -> {npz_path} ({size_mb:.1f} MB)")

    # ── SAVE METADATA ──────────────────────────────────────
    meta_out = OUTPUT_DIR / "metadata_parsed.csv"

    if all_records:
        with open(meta_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)

        print(f"Saved metadata -> {meta_out} ({len(all_records)} rows)")

    print(
        f"\nDone. {len(representations)} representations across "
        f"{len({r['speaker'] for r in all_records})} speakers, "
        f"{len({r['word'] for r in all_records})} words."
    )

    assert npz_path.exists(), f"[ERROR] NPZ was not saved to {npz_path}"
    assert meta_out.exists(), f"[ERROR] Metadata CSV was not saved to {meta_out}"
    print(f"\nrepresentations_float32.npz  saved at  {npz_path.resolve()}")
    print(f"metadata_parsed.csv saved at  {meta_out.resolve()}")
    print(f"SAVED")


if __name__ == "__main__":
    main()
