"""Read-only helpers for reconstructing (rollout_text, reward) pairs for
TTRL/EMPO/TTRV from LFRL_Baselines' training logs.

No rollout+advantage cache exists on disk for these methods (confirmed by
investigation): job_outputs/<run>_<jobid>.out has the raw completion texts
(one "completions: [...]" print per training step) but not the rewards;
checkpoints_lfrl_baselines/<method>/<run>/train_log.jsonl has the rewards
but not completion text. They are 1:1 by construction (same loop iteration
writes both, in the same order -- confirmed by reading train_ttrl.py/
train_ttrv.py directly: the reward is computed from the exact same
`completions` list that gets printed, with no reordering).

Runs can be resumed after preemption (many of ours were: 27 of 68 canonical
runs have >1 .out file). A resume reloads the last checkpoint and re-runs
steps forward from there, so train_log.jsonl step numbers can repeat across
files -- but empirically the total completions-line count across a run's
.out files always equals its train_log.jsonl line count exactly (verified:
18547 == 18547 for ttrl_full_flickr30k_test, its most-resumed case), so
concatenating .out files in mtime order and zipping positionally against
train_log.jsonl's natural (append) order is a valid 1:1 join -- this is
asserted at runtime, not just assumed.
"""
import ast
import json
import os
import re

COMPLETIONS_RE = re.compile(r"^\[train_(?:ttrl|ttrv)\] step \d+ completions: (\[.*\])\s*$")
OUT_FILE_RE_TMPL = r"^{}_(\d+)\.out$"

HEADER_RE_TTRL = re.compile(r"stage 1/2: TTRL training on \S+ \(K=(\d+), (\d+) epochs, voting=(\w+)")
HEADER_RE_TTRV = re.compile(
    r"stage 1/2: TTRV training on \S+ \(K=(\d+), (\d+) epochs, voting=(\w+), "
    r"temp=([\d.]+), lora_rank=(\d+), entropy_coef=([\d.]+)\)"
)
HEADER_RE_SRLM = re.compile(
    r"stage 1/2: SRLM training on \S+ \(K=(\d+) candidates, (\d+) epochs, "
    r"lora_rank=(\d+), self-judge \+ DPO\)"
)


def sorted_out_files(job_outputs_dir, run_name):
    """.out files for run_name, in chronological (mtime) order. Excludes
    lookalike names with a non-numeric suffix (e.g. run_name + '_evalonly_...'),
    which are separate eval-only jobs, not training continuations."""
    pat = re.compile(OUT_FILE_RE_TMPL.format(re.escape(run_name)))
    files = [f for f in os.listdir(job_outputs_dir) if pat.match(f)]
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(job_outputs_dir, f)))


def iter_completions(job_outputs_dir, run_name):
    for fname in sorted_out_files(job_outputs_dir, run_name):
        path = os.path.join(job_outputs_dir, fname)
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = COMPLETIONS_RE.match(line)
                if m:
                    yield ast.literal_eval(m.group(1))


def iter_train_log(train_log_path):
    with open(train_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_run_records(job_outputs_dir, train_log_path, run_name):
    """Returns a list of (train_log_entry, completions) tuples in chronological
    order, asserting the positional 1:1 join holds (see module docstring).

    Two different resume behaviors are observed across runs: some append to
    train_log.jsonl across restarts and partially catch up (total completions
    across ALL .out files == train_log line count), others overwrite
    train_log.jsonl fresh on every restart (only the most recent .out file's
    completions == train_log line count; earlier, abandoned attempts --
    sometimes even under a different hyperparameter, e.g. voting_mode changed
    between attempts -- are orphaned). Rather than assume either, try
    growing suffixes of the .out file list starting from the most recent
    file and stop at the first (smallest) suffix whose total completions
    count matches train_log's line count exactly."""
    entries = list(iter_train_log(train_log_path))
    files = sorted_out_files(job_outputs_dir, run_name)  # chronological, oldest first

    for start in range(len(files) - 1, -1, -1):
        suffix = files[start:]
        completions_list = []
        for fname in suffix:
            path = os.path.join(job_outputs_dir, fname)
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = COMPLETIONS_RE.match(line)
                    if m:
                        completions_list.append(ast.literal_eval(m.group(1)))
        if len(completions_list) == len(entries):
            return list(zip(entries, completions_list)), suffix

    raise ValueError(
        f"{run_name}: no suffix of {[f for f in files]} has a completions count "
        f"matching train_log's {len(entries)} lines -- positional join is NOT safe here"
    )


def find_canonical_entry(records, sample_id):
    """'Initial supervision before parameter updates' = the sample's
    minimum-epoch occurrence. If a resume re-ran epoch 1 for this sample_id
    more than once, take the chronologically LAST such occurrence -- that's
    the one whose resulting update actually survived into the persisted
    checkpoint trajectory (the earlier, pre-crash attempt's gradient step
    never got checkpointed). Returns (entry, completions, n_duplicates)."""
    matches = [(e, c) for e, c in records if e["sample_id"] == sample_id]
    if not matches:
        return None, None, 0
    min_epoch = min(e["epoch"] for e, c in matches)
    at_min_epoch = [(e, c) for e, c in matches if e["epoch"] == min_epoch]
    entry, completions = at_min_epoch[-1]
    return entry, completions, len(at_min_epoch)


def parse_ttrv_header(job_outputs_dir, fname):
    """fname: the specific .out file whose header to parse -- pass the
    OLDEST file of whichever suffix load_run_records() actually used, since
    hyperparameters (e.g. voting_mode) can differ between abandoned earlier
    attempts and the kept one."""
    with open(os.path.join(job_outputs_dir, fname), encoding="utf-8", errors="replace") as f:
        head = f.read(4000)
    m = HEADER_RE_TTRV.search(head)
    if not m:
        return None
    return {
        "k": int(m.group(1)),
        "epochs": int(m.group(2)),
        "voting_mode": m.group(3),
        "temperature": float(m.group(4)),
        "lora_rank": int(m.group(5)),
        "entropy_coef": float(m.group(6)),
    }


def parse_srlm_header(job_outputs_dir, fname):
    with open(os.path.join(job_outputs_dir, fname), encoding="utf-8", errors="replace") as f:
        head = f.read(4000)
    m = HEADER_RE_SRLM.search(head)
    if not m:
        return None
    return {"k": int(m.group(1)), "epochs": int(m.group(2)), "lora_rank": int(m.group(3))}
