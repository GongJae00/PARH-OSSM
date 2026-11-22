
import copy
import os
import pickle
import re

import numpy as np


def sanitize_trial_key(value: str, fallback: str = "trial") -> str:
	"""Normalize a trial key to filename-safe `[A-Za-z0-9._-]`."""
	raw = str(value or "").strip()
	if not raw:
		raw = fallback
	safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
	safe = safe.strip("._-")
	if not safe:
		safe = fallback
	return safe


def derive_trial_identifiers(sample: dict, dataset_name: str = "", sample_index: int = 0):
	"""Derive deterministic short/full trial identifiers.

	Returns:
		(short_key, full_key)
	"""
	dataset = str(
		(sample or {}).get("dataset_name")
		or (sample or {}).get("dataset")
		or (sample or {}).get("dataset_slug")
		or dataset_name
		or "unknown"
	).strip().lower()
	subject = str((sample or {}).get("subject", "")).strip()
	trial = str((sample or {}).get("trial", "")).strip()

	if subject and trial:
		short_key = sanitize_trial_key(f"{subject}_{trial}", fallback=f"idx_{sample_index}")
		full_key = sanitize_trial_key(f"{dataset}__{subject}__{trial}", fallback=short_key)
		return short_key, full_key
	if subject:
		short_key = sanitize_trial_key(subject, fallback=f"idx_{sample_index}")
		full_key = sanitize_trial_key(f"{dataset}__{subject}", fallback=short_key)
		return short_key, full_key

	video_path = str((sample or {}).get("video_path", "")).strip()
	if video_path:
		stem = os.path.splitext(os.path.basename(video_path))[0]
		short_key = sanitize_trial_key(stem, fallback=f"idx_{sample_index}")
		full_key = sanitize_trial_key(f"{dataset}__{stem}", fallback=short_key)
		return short_key, full_key

	short_key = sanitize_trial_key(f"idx_{sample_index}", fallback="idx_0")
	full_key = sanitize_trial_key(f"{dataset}__{short_key}", fallback=short_key)
	return short_key, full_key


def _sanitize_run_label(label):
	if not label:
		return None
	label = label.strip()
	if not label:
		return None
	sanitized = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in label)
	sanitized = sanitized.strip('_-')
	return sanitized or None


def _deep_merge_dict(base, new_values):
	if not isinstance(new_values, dict):
		return base
	for key, value in new_values.items():
		if isinstance(value, dict) and isinstance(base.get(key), dict):
			base[key] = _deep_merge_dict(base[key], value)
		else:
			base[key] = copy.deepcopy(value)
	return base


def _filter_valid_rois(rois):
	filtered = []
	for roi in rois or []:
		arr = np.asarray(roi)
		if arr.size > 0:
			filtered.append(roi)
	return filtered


def write_trial_results(out_path, results):
    payload = {key: results[key] for key in ('video_path', 'fps', 'gt', 'fs_gt', 'estimates')}
    with open(out_path, 'wb') as fp:
        pickle.dump(payload, fp)
