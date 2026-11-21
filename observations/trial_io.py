
import os
import hashlib
import numpy as np
import copy
import re
import contextlib
import tempfile
import pickle

def _dataset_results_dir(results_dir, dataset_name):
	path = os.path.join(results_dir, dataset_name)
	os.makedirs(path, exist_ok=True)
	# New structure for artifacts
	for sub in ("data", "metrics", "plots", "logs"):
		os.makedirs(os.path.join(path, sub), exist_ok=True)
	return path


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


FRAME_LOG_MANIFEST_SCHEMA = "frame_logs_manifest.v1"


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


def _method_token(method_name):
	name = method_name.strip()
	lname = name.lower()
	if lname in ('of_model', 'of', 'of_farneback'):
		return 'of_farneback'
	if lname in ('of_disp_bridge', 'of_displacement_bridge', 'of_bridge'):
		return 'of_disp_bridge'
	if lname == 'dof':
		return 'dof'
	if lname in ('dof_disp_bridge', 'dof_bridge', 'dof_displacement_bridge'):
		return 'dof_disp_bridge'
	if lname in ('profile1d_linear_bridge', 'profile1d_linear_displacement_bridge'):
		return 'profile1d_linear_bridge'
	if lname in ('profile1d_quadratic_bridge', 'profile1d_quadratic_displacement_bridge'):
		return 'profile1d_quadratic_bridge'
	if lname in ('profile1d_cubic_bridge', 'profile1d_cubic_displacement_bridge'):
		return 'profile1d_cubic_bridge'
	if lname in ('profile1d_consensus', 'profile1d_cons'):
		return 'profile1d_consensus'
	if lname.startswith('profile1d'):
		return lname.replace(' ', '_')
	if '__' in name:
		return name.replace(' ', '_')
	return name.replace(' ', '_').replace('-', '_')


def _method_suffix(methods):
	tokens = []
	seen = set()
	for m in methods:
		token = _method_token(m.name)
		if token in seen:
			continue
		seen.add(token)
		tokens.append(token)
	suffix = '_'.join(tokens)
	if len(suffix) <= 64:
		return suffix
	digest = hashlib.sha1(suffix.encode('utf-8')).hexdigest()[:10]
	return f"{len(tokens)}m_{digest}"


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

@contextlib.contextmanager
def _file_lock(lock_path):
	os.makedirs(os.path.dirname(lock_path), exist_ok=True)
	fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
	try:
		if os.name == 'nt':
			import msvcrt  # type: ignore
			os.lseek(fd, 0, os.SEEK_SET)
			msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
		else:
			import fcntl  # type: ignore
			fcntl.flock(fd, fcntl.LOCK_EX)
		yield
	finally:
		try:
			if os.name == 'nt':
				import msvcrt  # type: ignore
				os.lseek(fd, 0, os.SEEK_SET)
				msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
			else:
				import fcntl  # type: ignore
				fcntl.flock(fd, fcntl.LOCK_UN)
		finally:
			os.close(fd)


def _atomic_pickle_dump(data, path):
	dir_name = os.path.dirname(path)
	os.makedirs(dir_name, exist_ok=True)
	fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
	try:
		with os.fdopen(fd, 'wb') as tmp_file:
			pickle.dump(data, tmp_file)
		os.replace(tmp_path, path)
	except Exception:
		try:
			os.unlink(tmp_path)
		except OSError:
			pass
		raise




































def _filter_valid_rois(rois):
	filtered = []
	for roi in rois or []:
		arr = np.asarray(roi)
		if arr.size > 0:
			filtered.append(roi)
	return filtered


def _merge_results_payload(out_path, partial_results, method_order=None):
    if partial_results is None:
        return
    metadata_keys = ['video_path', 'fps', 'gt', 'fs_gt']
    new_estimates = list(partial_results.get('estimates', [])) if isinstance(partial_results, dict) else []
    lock_path = f"{out_path}.lock"
    with _file_lock(lock_path):
        if os.path.exists(out_path):
            with open(out_path, 'rb') as fp:
                data = pickle.load(fp)
        else:
            data = {}
        for key in metadata_keys:
            if key in partial_results:
                data[key] = partial_results[key]

        existing_estimates = list(data.get('estimates', []))
        # Keep only methods present in this run manifest to prevent stale
        # methods from previous runs leaking into current evaluation.
        if method_order:
            allowed = set(method_order)
            existing_estimates = [
                entry for entry in existing_estimates
                if isinstance(entry, dict) and entry.get('method') in allowed
            ]

        index_map = {}
        for idx, entry in enumerate(existing_estimates):
            if isinstance(entry, dict):
                name = entry.get('method')
                if name is not None and name not in index_map:
                    index_map[name] = idx

        for entry in new_estimates:
            if not isinstance(entry, dict):
                continue
            name = entry.get('method')
            if name is None:
                continue
            if name in index_map:
                existing_estimates[index_map[name]] = entry
            else:
                index_map[name] = len(existing_estimates)
                existing_estimates.append(entry)

        if method_order:
            order_index = {name: idx for idx, name in enumerate(method_order)}

            def _order_key(item):
                if isinstance(item, dict):
                    return order_index.get(item.get('method'), len(order_index))
                return len(order_index)

            existing_estimates.sort(key=_order_key)
        data['estimates'] = existing_estimates
        _atomic_pickle_dump(data, out_path)
    # Best-effort cleanup: keep lock file lifecycle short to avoid stale
    # *.pkl.lock artifacts after successful single-process runs.
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError:
        pass
