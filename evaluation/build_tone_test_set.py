"""
Build the custom tone test set for TTS evaluation.

Samples from the VieNeu validation filelist (which has ground truth audio)
and adds curated tone-specific test cases for targeted evaluation.

Usage:
    cd /home/bes/Desktop/TTS
    python Capstone_project/evaluation/build_tone_test_set.py

Output: Capstone_project/evaluation/tone_test_set.json
"""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '../..'))
_VITS2_DIR = os.path.join(_PROJECT_ROOT, 'vits2_pytorch')

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _VITS2_DIR not in sys.path:
    sys.path.insert(0, _VITS2_DIR)

from Capstone_project.tone_encoder.tone_utils import (
    cleaned_text_to_tone_sequence,
    extract_tones_per_position,
)

# Paths
VAL_FILELIST = os.path.join(_VITS2_DIR, 'filelists', 'vieneu_val_filelist.txt')
OUTPUT_PATH = os.path.join(_SCRIPT_DIR, 'tone_test_set.json')

# Target speakers for evaluation (diverse selection)
TARGET_SPEAKERS = [0, 5, 15, 19, 36, 42, 57, 76, 109, 143]

random.seed(42)


def load_val_data():
    """Load validation filelist entries."""
    entries = []
    with open(VAL_FILELIST, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 3:
                entries.append({
                    'audio_path': parts[0],
                    'speaker_id': int(parts[1]),
                    'ipa_text': parts[2],
                })
    return entries


def get_syllable_count(ipa_text):
    """Count syllables (non-empty non-punctuation tokens split by space)."""
    tokens = ipa_text.split()
    count = 0
    for t in tokens:
        cleaned = t.strip(',.!?;:')
        if cleaned:
            count += 1
    return count


def get_tone_distribution(ipa_text):
    """Get tone distribution from IPA text."""
    tones = extract_tones_per_position(ipa_text)
    dist = Counter(t for t in tones if t > 0)
    return dist


def has_tone(ipa_text, target_tone):
    """Check if text contains a specific tone."""
    tones = extract_tones_per_position(ipa_text)
    return target_tone in tones


def build_test_set():
    """Build the complete tone test set."""
    val_data = load_val_data()
    print(f"Loaded {len(val_data)} validation entries")

    # Index by speaker
    by_speaker = defaultdict(list)
    for entry in val_data:
        by_speaker[entry['speaker_id']].append(entry)

    test_set = []
    test_id = 0

    def add_entry(category, entry, notes=""):
        nonlocal test_id
        test_id += 1
        tone_seq = cleaned_text_to_tone_sequence(entry['ipa_text'])
        test_set.append({
            'id': f'tone_{test_id:04d}',
            'category': category,
            'ipa_text': entry['ipa_text'],
            'speaker_id': entry['speaker_id'],
            'expected_tones': tone_seq,
            'ground_truth_audio': entry['audio_path'],
            'notes': notes,
        })

    # --- Category 1: Sentences with tone 2 (sắc) dominant ---
    print("Selecting sắc-dominant samples...")
    sac_samples = [e for e in val_data if e['speaker_id'] in TARGET_SPEAKERS]
    sac_dominant = []
    for e in sac_samples:
        dist = get_tone_distribution(e['ipa_text'])
        if dist.get(2, 0) >= 3:
            sac_dominant.append(e)
    random.shuffle(sac_dominant)
    for e in sac_dominant[:25]:
        add_entry('tone_2_sac_dominant', e, 'sắc-dominant sentence')

    # --- Category 2: Sentences with tone 6 (nặng) dominant ---
    print("Selecting nặng-dominant samples...")
    nang_dominant = []
    for e in sac_samples:
        dist = get_tone_distribution(e['ipa_text'])
        if dist.get(6, 0) >= 3:
            nang_dominant.append(e)
    random.shuffle(nang_dominant)
    for e in nang_dominant[:25]:
        add_entry('tone_6_nang_dominant', e, 'nặng-dominant sentence')

    # --- Category 3: Sentences with tone 4 (hỏi) ---
    print("Selecting hỏi samples...")
    hoi_samples = [e for e in sac_samples if has_tone(e['ipa_text'], 4)]
    random.shuffle(hoi_samples)
    for e in hoi_samples[:25]:
        add_entry('tone_4_hoi', e, 'contains hỏi tone')

    # --- Category 4: Sentences with tone 5 (ngã) ---
    print("Selecting ngã samples...")
    nga_samples = [e for e in sac_samples if has_tone(e['ipa_text'], 5)]
    random.shuffle(nga_samples)
    for e in nga_samples[:25]:
        add_entry('tone_5_nga', e, 'contains ngã tone')

    # --- Category 5: Sentences with both hỏi and ngã (confusion pair) ---
    print("Selecting hỏi/ngã mixed samples...")
    hoi_nga_mix = [
        e for e in sac_samples
        if has_tone(e['ipa_text'], 4) and has_tone(e['ipa_text'], 5)
    ]
    random.shuffle(hoi_nga_mix)
    for e in hoi_nga_mix[:30]:
        add_entry('hoi_nga_mixed', e, 'contains both hỏi and ngã - confusion pair')

    # --- Category 6: Sentences with both sắc and nặng (confusion pair) ---
    print("Selecting sắc/nặng mixed samples...")
    sac_nang_mix = [
        e for e in sac_samples
        if has_tone(e['ipa_text'], 2) and has_tone(e['ipa_text'], 6)
    ]
    random.shuffle(sac_nang_mix)
    for e in sac_nang_mix[:30]:
        add_entry('sac_nang_mixed', e, 'contains both sắc and nặng - confusion pair')

    # --- Category 7: All 6 tones present in one sentence ---
    print("Selecting all-tones samples...")
    all_tones = []
    for e in sac_samples:
        dist = get_tone_distribution(e['ipa_text'])
        present = set(dist.keys())
        # At least 4 different tones (since 3 is missing, max possible is 5: 1,2,4,5,6)
        if len(present) >= 4:
            all_tones.append(e)
    random.shuffle(all_tones)
    for e in all_tones[:30]:
        dist = get_tone_distribution(e['ipa_text'])
        add_entry('multi_tone', e, f'tones present: {sorted(dist.keys())}')

    # --- Category 8: Short sentences (<5 syllables) ---
    print("Selecting short sentences...")
    short = [
        e for e in sac_samples
        if get_syllable_count(e['ipa_text']) <= 5
    ]
    random.shuffle(short)
    for e in short[:30]:
        add_entry('short_sentence', e, f'{get_syllable_count(e["ipa_text"])} syllables')

    # --- Category 9: Long sentences (>15 syllables) ---
    print("Selecting long sentences...")
    long_sent = [
        e for e in sac_samples
        if get_syllable_count(e['ipa_text']) >= 15
    ]
    random.shuffle(long_sent)
    for e in long_sent[:30]:
        add_entry('long_sentence', e, f'{get_syllable_count(e["ipa_text"])} syllables')

    # --- Category 10: Question sentences (ending with ?) ---
    print("Selecting question sentences...")
    questions = [e for e in sac_samples if e['ipa_text'].rstrip().endswith('?')]
    random.shuffle(questions)
    for e in questions[:20]:
        add_entry('question', e, 'question intonation')

    # --- Category 11: Multi-speaker same text ---
    print("Selecting multi-speaker samples...")
    # Find texts that appear for multiple speakers (from val set)
    text_to_entries = defaultdict(list)
    for e in val_data:
        text_to_entries[e['ipa_text']].append(e)
    # If same text doesn't appear for multiple speakers, use similar-length texts
    multi_spk_count = 0
    for text, entries in text_to_entries.items():
        speakers_in = set(e['speaker_id'] for e in entries)
        target_in = speakers_in & set(TARGET_SPEAKERS)
        if len(target_in) >= 2 and multi_spk_count < 30:
            for e in entries:
                if e['speaker_id'] in TARGET_SPEAKERS and multi_spk_count < 30:
                    add_entry(
                        'multi_speaker',
                        e,
                        f'same text, speaker {e["speaker_id"]}',
                    )
                    multi_spk_count += 1

    # --- Category 12: General coverage ---
    print("Selecting general coverage samples...")
    # Use samples from all target speakers not yet included
    used_audios = {e['ground_truth_audio'] for e in test_set}
    general = [
        e for e in val_data
        if e['speaker_id'] in TARGET_SPEAKERS
        and e['audio_path'] not in used_audios
    ]
    random.shuffle(general)
    for e in general[:40]:
        add_entry('general_coverage', e, 'diverse coverage')

    # Summary
    print(f"\n{'='*60}")
    print(f"Total test set size: {len(test_set)}")
    cat_counts = Counter(e['category'] for e in test_set)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    spk_counts = Counter(e['speaker_id'] for e in test_set)
    print(f"\nSpeakers represented: {len(spk_counts)}")
    for spk, count in sorted(spk_counts.items()):
        print(f"  Speaker {spk}: {count} samples")

    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {OUTPUT_PATH}")

    return test_set


if __name__ == '__main__':
    build_test_set()
