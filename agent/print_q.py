import argparse
import csv
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from rl import DQN, Q
from utils import setup_logger

load_dotenv()
logger = setup_logger(
    "kubernetes_agent", log_level=os.getenv("LOG_LEVEL", "INFO"), log_to_file=True
)


def build_agent(logger):
    """Construct agent instance from environment variables (same logic as before)."""
    choose_algorithm = os.getenv("ALGORITHM", "Q").upper()
    start_time = int(time.time())
    if choose_algorithm == "Q":
        agent = Q(
            learning_rate=float(os.getenv("LEARNING_RATE", "0.1")),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", "0.95")),
            epsilon_start=0.0,
            epsilon_decay=0.0,
            epsilon_min=0.0,
            created_at=start_time,
            logger=logger,
        )
    elif choose_algorithm == "DQN":
        agent = DQN(
            learning_rate=float(os.getenv("LEARNING_RATE", "0.001")),
            discount_factor=float(os.getenv("DISCOUNT_FACTOR", "0.95")),
            epsilon_start=0.0,
            epsilon_decay=0.0,
            epsilon_min=0.0,
            device=os.getenv("DEVICE", "cpu"),
            buffer_size=int(os.getenv("BUFFER_SIZE", "50000")),
            batch_size=int(os.getenv("BATCH_SIZE", "64")),
            target_update_freq=int(os.getenv("TARGET_UPDATE_FREQ", "100")),
            grad_clip_norm=float(os.getenv("GRAD_CLIP_NORM", "10.0")),
            created_at=start_time,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {choose_algorithm}")
    return agent


def format_state(state_key) -> str:
    """Turn the state tuple into a readable string.

    Expected state shape (cpu, memory, response_time, last_action).
    """
    try:
        cpu, memory, response_time, last_action = state_key
        return (
            f"cpu={cpu}, mem={memory}, resp={response_time}, last_action={last_action}"
        )
    except Exception:
        # Fallback to repr
        return repr(state_key)


def top_actions_str(q_vals: np.ndarray, top_k: int) -> str:
    """Return a compact string of top_k actions and values: idx:val,..."""
    if not isinstance(q_vals, np.ndarray):
        q_vals = np.array(q_vals)
    if q_vals.size == 0:
        return "[]"
    idxs = np.argsort(q_vals)[::-1][:top_k]
    pairs = [f"{int(i)}:{float(q_vals[int(i)]):.4f}" for i in idxs]
    return "[" + ", ".join(pairs) + "]"


def print_q_table(
    agent, top_k: int = 5, limit: Optional[int] = 50, output_csv: Optional[str] = None
):
    """Pretty-print the Q-table from the agent.

    - top_k: number of top actions to show per state
    - limit: max number of state rows to print (sorted by best Q value desc)
    - output_csv: optional path to save CSV with the printed rows
    """
    q_table = getattr(agent, "q_table", None)
    if not q_table:
        # Try to load q_table directly from the model file (pickle) as a fallback
        logger.info(
            "Agent has no q_table or it is empty — attempting direct model file inspection"
        )
        return False

    rows = []
    for state_key, q_vals in q_table.items():
        try:
            arr = np.array(q_vals)
        except Exception:
            arr = np.array(list(q_vals))
        if arr.size == 0:
            best_idx = None
            best_val = float("nan")
        else:
            best_idx = int(np.argmax(arr))
            best_val = float(np.max(arr))

        rows.append(
            {
                "state_key": state_key,
                "state_str": format_state(state_key),
                "best_action": best_idx,
                "best_value": best_val,
                "top_actions": top_actions_str(arr, top_k),
            }
        )

    # Sort rows by best_value (descending)
    rows.sort(
        key=lambda r: (
            float(r["best_value"]) if r["best_value"] is not None else float("-inf")
        ),
        reverse=True,
    )

    rows_to_print = rows[:limit] if limit is not None else rows

    header = f"{'STATE':<45} {'BEST_A':>6} {'BEST_Q':>10} {'TOP_ACTIONS'}"
    print(header)
    print("-" * (len(header) + 30))
    for r in rows_to_print:
        state_display = (
            (r["state_str"][:42] + "...")
            if len(r["state_str"]) > 45
            else r["state_str"]
        )
        best_a = r["best_action"] if r["best_action"] is not None else "-"
        _bv = r["best_value"]
        if isinstance(_bv, float) and not np.isnan(_bv):
            best_q = f"{_bv:.4f}"
        else:
            best_q = "-"
        print(f"{state_display:<45} {best_a:>6} {best_q:>10} {r['top_actions']}")

    logger.info(f"Printed {len(rows_to_print)} of {len(rows)} states from Q-table")

    if output_csv:
        try:
            with Path(output_csv).open("w", newline="") as csvfile:
                fieldnames = [
                    "cpu",
                    "memory",
                    "response_time",
                    "last_action",
                    "best_action",
                    "best_value",
                    "top_actions",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows_to_print:
                    sk = r["state_key"]
                    if isinstance(sk, (list, tuple)) and len(sk) >= 4:
                        cpu, memory, response_time, last_action = sk[:4]
                    else:
                        # Fallback to putting full state in cpu column
                        cpu, memory, response_time, last_action = (repr(sk), "", "", "")
                    writer.writerow(
                        {
                            "cpu": cpu,
                            "memory": memory,
                            "response_time": response_time,
                            "last_action": last_action,
                            "best_action": r["best_action"],
                            "best_value": r["best_value"],
                            "top_actions": r["top_actions"],
                        }
                    )
            logger.info(f"Saved Q-table excerpt to CSV: {output_csv}")
        except Exception as e:
            logger.error(f"Failed to write CSV to {output_csv}: {e}")
    return True


def try_load_q_from_file(path: str):
    """Attempt to extract a q_table from the model file directly.

    Returns a tuple (q_table, info) where q_table is the dict or None and info is a
    short description string for logging.
    """
    p = Path(path)
    if not p.exists():
        return None, f"Model file not found: {path}"

    # Try pickle first (Q.save_model uses pickle)
    try:
        with p.open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "q_table" in data:
            return data["q_table"], "Loaded q_table from pickle checkpoint"
        # If the file itself is a q_table mapping
        if isinstance(data, dict) and all(isinstance(k, tuple) for k in data.keys()):
            return data, "Loaded q_table mapping directly from pickle"
    except Exception:
        # Not a pickle or failed to parse — continue
        pass

    # Try torch load for .pth/.pt files — optional import
    try:
        import torch

        data = torch.load(str(p), map_location="cpu")
        if isinstance(data, dict):
            # Common checkpoint might store q_table inside
            if "q_table" in data:
                return data["q_table"], "Loaded q_table from torch checkpoint dict"
            # Some checkpoints store nested checkpoints
            for v in data.values():
                if isinstance(v, dict) and "q_table" in v:
                    return v["q_table"], "Loaded nested q_table from torch checkpoint"
        # If nothing found, just return None but include info about keys
        keys = list(data.keys()) if isinstance(data, dict) else []
        return None, f"Torch checkpoint loaded but no q_table key found. Keys: {keys}"
    except Exception as e:
        return None, f"Failed to inspect model file with torch/pickle: {e}"


def _parse_args():
    p = argparse.ArgumentParser(description="Print readable Q-table from saved model")
    p.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_PATH", "model/dqn/1697041234_default/best_model.pth"),
        help="Path to the saved model file (pickle)",
    )
    p.add_argument(
        "--top-k", type=int, default=5, help="How many top actions to show per state"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of state rows to print (use 0 or -1 to print all)",
    )
    p.add_argument(
        "--output-csv", type=str, default=None, help="Optional path to save CSV output"
    )
    p.add_argument(
        "--states-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV file with states to evaluate against a DQN policy_net. "
            "Columns: cpu,memory,response_time,last_action"
        ),
    )
    return p.parse_args()


def evaluate_states(
    agent, states: List[tuple], top_k: int = 5, output_csv: Optional[str] = None
):
    """Evaluate a list of states with a DQN policy_net and print top-k actions per state.

    Assumes each state is a tuple/list in order: (cpu, memory, response_time, last_action).
    This mirrors the get_state_key ordering used by the Q agent. If your DQN uses a different
    preprocessing or normalization, results may need adjustment.
    """
    if not hasattr(agent, "policy_net"):
        logger.error("Agent has no policy_net to evaluate states")
        return False

    # Build numpy array
    try:
        arr = np.array(states, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to build state array: {e}")
        return False

    try:
        agent.policy_net.eval()
        with torch.no_grad():
            t = torch.from_numpy(arr)
            outputs = agent.policy_net(t)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            out_np = outputs.cpu().numpy()
    except Exception as e:
        logger.error(f"Failed to run policy_net on states: {e}")
        return False

    rows = []
    for s, scores in zip(states, out_np):
        idxs = np.argsort(scores)[::-1][:top_k]
        pairs = [f"{int(i)}:{float(scores[int(i)]):.4f}" for i in idxs]
        rows.append(
            {
                "state": s,
                "top_actions": "[" + ", ".join(pairs) + "]",
                "top_indices": ",".join(str(int(i)) for i in idxs),
                "top_values": ",".join(f"{float(scores[int(i)]):.4f}" for i in idxs),
            }
        )

    # Print
    header = f"{'STATE':<45} {'TOP_ACTIONS'}"
    print(header)
    print("-" * (len(header) + 10))
    for r in rows:
        state_str = (
            format_state(r["state"]) if not isinstance(r["state"], str) else r["state"]
        )
        state_display = (state_str[:42] + "...") if len(state_str) > 45 else state_str
        print(f"{state_display:<45} {r['top_actions']}")

    if output_csv:
        try:
            with Path(output_csv).open("w", newline="") as csvfile:
                fieldnames = [
                    "cpu",
                    "memory",
                    "response_time",
                    "last_action",
                    "top_indices",
                    "top_values",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    sk = r["state"]
                    if isinstance(sk, (list, tuple)) and len(sk) >= 4:
                        cpu, memory, response_time, last_action = sk[:4]
                    else:
                        cpu, memory, response_time, last_action = (repr(sk), "", "", "")
                    writer.writerow(
                        {
                            "cpu": cpu,
                            "memory": memory,
                            "response_time": response_time,
                            "last_action": last_action,
                            "top_indices": r["top_indices"],
                            "top_values": r["top_values"],
                        }
                    )
            logger.info(f"Saved state evaluation to CSV: {output_csv}")
        except Exception as e:
            logger.error(f"Failed to write CSV for state evaluation: {e}")

    return True


if __name__ == "__main__":
    args = _parse_args()
    model_path = args.model_path
    logger.info(f"Loading trained model from: {model_path}")
    agent = build_agent(logger)
    try:
        agent.load_model(model_path)
    except Exception as exc:
        logger.error(f"Unable to load model {model_path}: {exc}")
        raise

    limit = None if args.limit <= 0 else args.limit
    printed = print_q_table(
        agent, top_k=args.top_k, limit=limit, output_csv=args.output_csv
    )
    if not printed:
        # Try direct file-based q_table extraction (pickle/torch)
        q_table, info = try_load_q_from_file(model_path)
        logger.info(info)
        if q_table:
            # Attach q_table to agent and retry printing
            try:
                agent.q_table = q_table
                logger.info("Attached q_table loaded from file to agent and printing")
                print_q_table(
                    agent, top_k=args.top_k, limit=limit, output_csv=args.output_csv
                )
            except Exception as e:
                logger.error(f"Failed to set q_table on agent: {e}")
        # If this is a DQN model, provide a brief summary of the policy_net output layer
        elif hasattr(agent, "policy_net"):
            try:
                net = agent.policy_net
                out_layer = None
                # Common final layer names
                for name in ("fc_out", "out", "fc3", "fc_final"):
                    if hasattr(net, name):
                        out_layer = getattr(net, name)
                        break
                if out_layer is None:
                    # Fallback: try to find last linear-like module
                    modules = [
                        m
                        for m in net.modules()
                        if hasattr(m, "weight") and hasattr(m, "bias")
                    ]
                    out_layer = modules[-1] if modules else None

                if out_layer is not None:
                    w = out_layer.weight.detach().cpu().numpy()
                    b = (
                        out_layer.bias.detach().cpu().numpy()
                        if hasattr(out_layer, "bias")
                        else None
                    )
                    topk = args.top_k
                    if b is not None:
                        idxs = np.argsort(b)[::-1][:topk]
                        logger.info(
                            f"Top {topk} actions by output bias: {idxs.tolist()}"
                        )
                        for i in idxs:
                            logger.info(f"Action {int(i)} bias={float(b[int(i)]):.4f}")
                    else:
                        # Score by L2 norm of weight columns
                        norms = np.linalg.norm(w, axis=1)
                        idxs = np.argsort(norms)[::-1][:topk]
                        logger.info(
                            f"Top {topk} actions by output weight norm: {idxs.tolist()}"
                        )
                        for i in idxs:
                            logger.info(
                                f"Action {int(i)} norm={float(norms[int(i)]):.4f}"
                            )
                else:
                    logger.error(
                        "Unable to find policy network output layer to summarize DQN checkpoint"
                    )
            except Exception as e:
                logger.error(f"Unable to summarize DQN policy_net: {e}")
        else:
            logger.error(
                "No q_table found and agent has no policy_net; nothing to print"
            )
