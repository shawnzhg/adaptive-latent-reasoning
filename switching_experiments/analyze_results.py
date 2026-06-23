"""
Step 5: Compare evaluation results between Group A (no anchor) and Group B (anchor).
Generates summary statistics and comparison table.
"""

import os, json, argparse, sys
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("analysis")


def load_results(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./anchor_data/eval_results")
    parser.add_argument("--baseline_label", type=str, default="baseline")
    parser.add_argument("--no_anchor_label", type=str, default="no_anchor")
    parser.add_argument("--anchor_label", type=str, default="anchor_0.3")
    args = parser.parse_args()

    labels = []
    all_results = {}

    for label in [args.baseline_label, args.no_anchor_label, args.anchor_label]:
        path = os.path.join(args.results_dir, f"eval_{label}.json")
        if os.path.exists(path):
            all_results[label] = load_results(path)
            labels.append(label)
        else:
            logger.warning(f"  Missing: {path}")

    if not labels:
        logger.error("No results found!")
        return

    # Get K values from first result
    K_values = sorted([int(k) for k in all_results[labels[0]].keys()
                        if k.isdigit()])

    # === Table 1: Accuracy vs K ===
    logger.info("\n" + "=" * 70)
    logger.info("TABLE 1: GSM8K Accuracy vs Latent Steps K")
    logger.info("=" * 70)

    header = f"{'K':>4}"
    for label in labels:
        header += f" | {label:>15}"
    logger.info(header)
    logger.info("-" * len(header))

    for K in K_values:
        row = f"{K:>4}"
        for label in labels:
            r = all_results[label]
            if str(K) in r:
                acc = r[str(K)]["accuracy"]
                row += f" | {acc:>14.1%}"
            else:
                row += f" | {'N/A':>15}"
        logger.info(row)

    # === Table 2: Exit Log-Likelihood vs K ===
    logger.info("\n" + "=" * 70)
    logger.info("TABLE 2: Exit Token Log-Likelihood vs K")
    logger.info("=" * 70)

    header = f"{'K':>4}"
    for label in labels:
        header += f" | {label:>15}"
    logger.info(header)
    logger.info("-" * len(header))

    for K in K_values:
        row = f"{K:>4}"
        for label in labels:
            r = all_results[label]
            if str(K) in r:
                ll = r[str(K)].get("avg_exit_loglik", 0)
                row += f" | {ll:>15.3f}"
            else:
                row += f" | {'N/A':>15}"
        logger.info(row)

    # === Table 3: Anchor Max Prob vs K ===
    logger.info("\n" + "=" * 70)
    logger.info("TABLE 3: LM Head Max Prob at Latent Steps (anchor quality)")
    logger.info("=" * 70)

    header = f"{'K':>4}"
    for label in labels:
        header += f" | {label:>15}"
    logger.info(header)
    logger.info("-" * len(header))

    for K in K_values:
        if K == 0:
            continue
        row = f"{K:>4}"
        for label in labels:
            r = all_results[label]
            if str(K) in r:
                prob = r[str(K)].get("avg_anchor_max_prob", 0)
                row += f" | {prob:>15.3f}"
            else:
                row += f" | {'N/A':>15}"
        logger.info(row)

    # === KV Similarity if available ===
    has_kv = any("kv_similarity" in all_results[l] for l in labels)
    if has_kv:
        logger.info("\n" + "=" * 70)
        logger.info("TABLE 4: KV-Cache Cosine Similarity vs K")
        logger.info("=" * 70)
        header = f"{'K':>4}"
        for label in labels:
            header += f" | {label:>15}"
        logger.info(header)
        logger.info("-" * len(header))
        for K in K_values:
            row = f"{K:>4}"
            for label in labels:
                r = all_results[label]
                kv = r.get("kv_similarity", {})
                if str(K) in kv:
                    row += f" | {kv[str(K)]:>15.4f}"
                else:
                    row += f" | {'N/A':>15}"
            logger.info(row)

    # === Verdict ===
    logger.info("\n" + "=" * 70)
    logger.info("VERDICT")
    logger.info("=" * 70)

    if args.no_anchor_label in all_results and args.anchor_label in all_results:
        r_a = all_results[args.no_anchor_label]
        r_b = all_results[args.anchor_label]

        # Find the K where accuracy drops below 60% for each
        def find_max_k(results, threshold=0.60):
            max_k = 0
            for K in K_values:
                if str(K) in results and results[str(K)]["accuracy"] >= threshold:
                    max_k = K
            return max_k

        max_k_a = find_max_k(r_a)
        max_k_b = find_max_k(r_b)

        logger.info(f"Max K where Acc >= 60%:")
        logger.info(f"  Group A (no anchor):   K = {max_k_a}")
        logger.info(f"  Group B (with anchor): K = {max_k_b}")
        logger.info(f"  Improvement: +{max_k_b - max_k_a} steps")

        if max_k_b > max_k_a:
            logger.info("\n✅ SEMANTIC ANCHORING IS EFFECTIVE.")
            logger.info(f"   It extends the latent reasoning horizon by {max_k_b - max_k_a} steps.")
        elif max_k_b == max_k_a:
            logger.info("\n⚠️  SEMANTIC ANCHORING HAS NO SIGNIFICANT EFFECT.")
            # Check if accuracy at same K is meaningfully higher
            for K in K_values:
                if K > 0 and str(K) in r_a and str(K) in r_b:
                    diff = r_b[str(K)]["accuracy"] - r_a[str(K)]["accuracy"]
                    if diff > 0.05:
                        logger.info(f"   However, at K={K}, anchor improves accuracy by {diff:.1%}")
        else:
            logger.info("\n❌ SEMANTIC ANCHORING MAY BE HARMFUL.")
            logger.info("   The anchor constraint may be too restrictive.")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()