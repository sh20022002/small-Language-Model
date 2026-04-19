"""
Semantic evaluation suite — standard industry benchmarks for small language models.

Tests included
--------------
1. Perplexity       WikiText-2 test set  (gold standard for LM quality)
2. BPC              bits per character   (compression proxy)
3. Top-k accuracy   next-token prediction accuracy (top-1 and top-5)
4. BLiMP            grammaticality judgment — 67 linguistic phenomena
                    (each has a grammatical vs ungrammatical pair; model should
                     assign lower loss to the grammatical sentence)
5. LAMBADA          0-shot last-word prediction
                    (requires understanding long-range context)
6. Embedding analogy  word2vec-style a:b :: c:? test on the embedding matrix

Expected results for a small ~10M-param model trained on WikiText-2:
  Perplexity       ~50–150  (GPT-2 small = 29.4, random = vocab_size)
  BPC              ~1.5–2.5 (GPT-2 small ≈ 1.0)
  Top-1 accuracy   ~20–35%
  Top-5 accuracy   ~45–60%
  BLiMP            ~55–65%  (chance = 50%, GPT-2 large ≈ 81%)
  LAMBADA          ~1–10%   (GPT-2 small ≈ 45%, hard task)
  Analogy          ~5–20%   (depends heavily on training data)

Usage
-----
# From Colab:
%run /content/small-Language-Model/tests/semantic_eval.py \\
     --model  /content/models/slm_checkpoint.pt \\
     --tok    /content/tokenizer.pkl.gz

# From command line:
python tests/semantic_eval.py --model models/slm.pt --tok tokenizer.pkl.gz --device cuda
"""

import argparse, math, sys, time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _encode(tok, text: str, max_len: Optional[int] = None) -> List[int]:
    """Encode text with either HybridTokenizer or a HuggingFace tokenizer."""
    if hasattr(tok, 'encode') and hasattr(tok, 'token2id'):
        # HybridTokenizer
        ids = tok.encode(text, mode='flat')
    else:
        # HuggingFace tokenizer fallback
        ids = tok.encode(text)
    return ids[:max_len] if max_len else ids


def _decode(tok, ids: List[int]) -> str:
    if hasattr(tok, 'token2id'):
        return tok.decode(ids)
    return tok.decode(ids, skip_special_tokens=True)


def _pad_id(tok) -> int:
    if hasattr(tok, 'token2id'):
        return tok.token2id.get('<PAD>', 0)
    return tok.pad_token_id or 0


@torch.no_grad()
def _batch_cross_entropy(
    model,
    token_ids: List[int],
    device: torch.device,
    stride: int = 128,
    max_len: int = 256,
) -> float:
    """Compute average cross-entropy loss over a token sequence using a sliding window."""
    if len(token_ids) < 2:
        return float('nan')

    total_loss, total_tokens = 0.0, 0

    for start in range(0, len(token_ids) - 1, stride):
        chunk = token_ids[start : start + max_len + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(chunk[1:],  dtype=torch.long).unsqueeze(0).to(device)

        logits = model(x)                           # [1, T, V]
        T = logits.size(1)
        loss = F.cross_entropy(
            logits.view(T, -1),
            y.view(T),
            reduction='sum',
        )
        total_loss   += loss.item()
        total_tokens += T

    return total_loss / max(total_tokens, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Perplexity + BPC
# ──────────────────────────────────────────────────────────────────────────────

def eval_perplexity(model, tok, device, n_examples: int = 500) -> dict:
    """Perplexity and bits-per-character on WikiText-2 test set."""
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    except Exception as e:
        return {'error': str(e)}

    model.eval()
    texts = [ex['text'] for ex in ds if len(ex['text'].strip()) > 30][:n_examples]

    total_ce, total_chars = 0.0, 0
    for text in texts:
        ids = _encode(tok, text)
        if len(ids) < 2:
            continue
        ce = _batch_cross_entropy(model, ids, device)
        total_ce   += ce * (len(ids) - 1)
        total_chars += len(text)

    total_tokens = sum(len(_encode(tok, t)) - 1 for t in texts if len(_encode(tok, t)) >= 2)
    avg_ce  = total_ce / max(total_tokens, 1)
    ppl     = math.exp(min(avg_ce, 20))          # cap to avoid overflow
    bpc     = avg_ce / math.log(2)               # convert nats → bits

    return {
        'perplexity': round(ppl, 2),
        'bpc':        round(bpc, 4),
        'n_texts':    len(texts),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Top-k next-token accuracy
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_topk_accuracy(model, tok, device, k_values=(1, 5), n_examples: int = 300) -> dict:
    """Fraction of next-token predictions where truth is in top-k."""
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    except Exception as e:
        return {'error': str(e)}

    model.eval()
    texts = [ex['text'] for ex in ds if len(ex['text'].strip()) > 30][:n_examples]

    correct  = {k: 0 for k in k_values}
    total    = 0
    max_len  = getattr(model, 'max_seq_len', 256)

    for text in texts:
        ids = _encode(tok, text, max_len=max_len + 1)
        if len(ids) < 2:
            continue
        x      = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
        labels = torch.tensor(ids[1:],  dtype=torch.long).to(device)

        logits  = model(x)                          # [1, T, V]
        logits  = logits.squeeze(0)                  # [T, V]

        for t in range(logits.size(0)):
            top_k_ids = logits[t].topk(max(k_values)).indices
            for k in k_values:
                if labels[t].item() in top_k_ids[:k].tolist():
                    correct[k] += 1
            total += 1

    return {
        f'top{k}_acc': round(correct[k] / max(total, 1) * 100, 2)
        for k in k_values
    } | {'n_predictions': total}


# ──────────────────────────────────────────────────────────────────────────────
# 3. BLiMP — grammaticality judgment
# ──────────────────────────────────────────────────────────────────────────────

# 10 representative BLiMP phenomena (out of 67 total)
BLIMP_PHENOMENA = [
    'anaphor_gender_agreement',
    'animate_subject_trans',
    'determiner_noun_agreement_1',
    'irregular_verb_form',
    'npi_present_1',
    'principle_A_domain_1',
    'sentential_negation_npi_scope',
    'subject_verb_agreement_1',
    'wh_questions_object_gap',
    'wh_questions_subject_gap',
]

@torch.no_grad()
def eval_blimp(model, tok, device, n_per_phenomenon: int = 100) -> dict:
    """
    For each pair (good_sentence, bad_sentence), check if the model assigns
    lower cross-entropy to the grammatical one. Accuracy = fraction correct.
    Chance = 50%.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        return {'error': str(e)}

    model.eval()
    results = {}
    overall_correct, overall_total = 0, 0

    for phenomenon in BLIMP_PHENOMENA:
        try:
            ds = load_dataset(
                'nyu-mll/blimp', phenomenon,
                split='train',
                trust_remote_code=True,
            ).select(range(min(n_per_phenomenon, 1000)))
        except Exception as e:
            results[phenomenon] = f'skip ({e})'
            continue

        correct, total = 0, 0
        for ex in ds:
            good = ex.get('sentence_good', '')
            bad  = ex.get('sentence_bad',  '')
            if not good or not bad:
                continue

            ids_good = _encode(tok, good)
            ids_bad  = _encode(tok, bad)
            if len(ids_good) < 2 or len(ids_bad) < 2:
                continue

            ce_good = _batch_cross_entropy(model, ids_good, device)
            ce_bad  = _batch_cross_entropy(model, ids_bad,  device)

            if ce_good < ce_bad:   # model prefers the grammatical sentence
                correct += 1
            total += 1

        acc = correct / max(total, 1) * 100
        results[phenomenon] = round(acc, 1)
        overall_correct += correct
        overall_total   += total

    results['overall'] = round(overall_correct / max(overall_total, 1) * 100, 2)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. LAMBADA — last-word prediction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_lambada(model, tok, device, n_examples: int = 500) -> dict:
    """
    Given a passage, predict the final word exactly (top-1) or within top-5.
    This task tests long-range context understanding.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset('EleutherAI/lambada_openai', split='test').select(range(n_examples))
    except Exception as e:
        return {'error': str(e)}

    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0

    for ex in ds:
        text = ex.get('text', '')
        if not text:
            continue

        # Split into context (all but last word) and target (last word)
        words = text.rsplit(' ', 1)
        if len(words) != 2:
            continue
        context, target_word = words
        context = context.strip()
        target_word = target_word.strip().lower()

        # Encode context
        ctx_ids = _encode(tok, context)
        if not ctx_ids:
            continue

        x      = torch.tensor(ctx_ids, dtype=torch.long).unsqueeze(0).to(device)
        logits = model(x)                           # [1, T, V]
        last_logits = logits[0, -1, :]               # [V]

        # Get top-5 predicted tokens and decode them
        top5_ids = last_logits.topk(5).indices.tolist()
        top5_words = [_decode(tok, [tid]).strip().lower() for tid in top5_ids]

        if top5_words and top5_words[0] == target_word:
            top1_correct += 1
        if target_word in top5_words:
            top5_correct += 1
        total += 1

    return {
        'top1_acc': round(top1_correct / max(total, 1) * 100, 2),
        'top5_acc': round(top5_correct / max(total, 1) * 100, 2),
        'n_examples': total,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Embedding analogy
# ──────────────────────────────────────────────────────────────────────────────

# Classic word2vec analogy pairs: a is to b as c is to ? (answer = d)
# Format: (a, b, c, d)
SEMANTIC_ANALOGIES: List[Tuple[str, str, str, str]] = [
    # Capital cities
    ('paris',  'france',  'berlin', 'germany'),
    ('paris',  'france',  'rome',   'italy'),
    ('paris',  'france',  'tokyo',  'japan'),
    ('london', 'england', 'berlin', 'germany'),
    # Gender
    ('king',   'queen',   'man',    'woman'),
    ('king',   'queen',   'brother','sister'),
    ('man',    'woman',   'boy',    'girl'),
    ('actor',  'actress', 'waiter', 'waitress'),
    # Verb tense
    ('walking','walked',  'running','ran'),
    ('walking','walked',  'swimming','swam'),
    ('go',     'went',    'see',    'saw'),
    # Comparatives
    ('good',   'better',  'bad',    'worse'),
    ('big',    'bigger',  'small',  'smaller'),
    # Plural
    ('cat',    'cats',    'dog',    'dogs'),
    ('city',   'cities',  'country','countries'),
]

def eval_embedding_analogy(model, tok, device) -> dict:
    """
    a:b :: c:?  — compute d = emb(b) - emb(a) + emb(c), find nearest token.
    Uses the model's token embedding matrix directly.
    """
    emb_matrix = model.token_emb.weight.detach().to(device)  # [V, D]
    # L2-normalise each row for cosine similarity
    norms      = emb_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    emb_norm   = emb_matrix / norms                           # [V, D]

    def get_emb(word: str) -> Optional[torch.Tensor]:
        ids = _encode(tok, word)
        if not ids:
            return None
        return emb_norm[ids[0]]   # embedding of first token of the word

    correct, total, skipped = 0, 0, 0

    for a, b, c, expected_d in SEMANTIC_ANALOGIES:
        ea, eb, ec = get_emb(a), get_emb(b), get_emb(c)
        if ea is None or eb is None or ec is None:
            skipped += 1
            continue

        # Analogy vector: direction from a to b, applied to c
        query = eb - ea + ec                           # [D]
        query = query / query.norm().clamp(min=1e-8)   # normalise

        sims = emb_norm @ query                        # [V] cosine similarities
        # Exclude a, b, c themselves from candidates
        for w in (a, b, c):
            ids = _encode(tok, w)
            if ids:
                sims[ids[0]] = -1.0

        pred_id   = sims.argmax().item()
        pred_word = _decode(tok, [pred_id]).strip().lower()

        expected_ids = _encode(tok, expected_d)
        expected_tok = _decode(tok, [expected_ids[0]]).strip().lower() if expected_ids else expected_d

        if pred_word == expected_tok or pred_word == expected_d:
            correct += 1
        total += 1

    return {
        'accuracy':   round(correct / max(total, 1) * 100, 2),
        'correct':    correct,
        'total':      total,
        'skipped':    skipped,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all(model, tok, device, quick: bool = False) -> dict:
    """Run the full evaluation suite. Set quick=True to use smaller sample sizes."""
    scale = 0.2 if quick else 1.0

    results = {}
    timings = {}

    suite = [
        ('perplexity_bpc',     eval_perplexity,       dict(n_examples=int(500 * scale))),
        ('topk_accuracy',      eval_topk_accuracy,    dict(n_examples=int(300 * scale))),
        ('blimp',              eval_blimp,             dict(n_per_phenomenon=int(100 * scale))),
        ('lambada',            eval_lambada,           dict(n_examples=int(500 * scale))),
        ('embedding_analogy',  eval_embedding_analogy, {}),
    ]

    for name, fn, kwargs in suite:
        print(f'\n  Running {name}...', end=' ', flush=True)
        t0 = time.time()
        try:
            res = fn(model, tok, device, **kwargs)
        except Exception as e:
            res = {'error': str(e)}
        elapsed = time.time() - t0
        results[name] = res
        timings[name] = round(elapsed, 1)
        print(f'done ({elapsed:.1f}s)')

    return {'results': results, 'timings_s': timings}


def print_report(report: dict) -> None:
    results = report['results']
    timings = report['timings_s']

    print('\n' + '=' * 60)
    print('  SEMANTIC EVALUATION REPORT')
    print('=' * 60)

    # 1. Perplexity + BPC
    r = results.get('perplexity_bpc', {})
    print(f'\n1. Perplexity & BPC (WikiText-2 test, n={r.get("n_texts","?")})')
    if 'error' in r:
        print(f'   ERROR: {r["error"]}')
    else:
        ppl = r.get('perplexity', '?')
        bpc = r.get('bpc', '?')
        ppl_note = '(lower is better; GPT-2-small=29.4, random≈vocab_size)'
        bpc_note = '(lower is better; GPT-2-small≈1.0, random≈16)'
        print(f'   Perplexity : {ppl}   {ppl_note}')
        print(f'   BPC        : {bpc}   {bpc_note}')

    # 2. Top-k accuracy
    r = results.get('topk_accuracy', {})
    print(f'\n2. Next-Token Accuracy (WikiText-2 val, n={r.get("n_predictions","?")})')
    if 'error' in r:
        print(f'   ERROR: {r["error"]}')
    else:
        print(f'   Top-1 : {r.get("top1_acc","?")}%   (GPT-2-small ≈ 35%)')
        print(f'   Top-5 : {r.get("top5_acc","?")}%   (GPT-2-small ≈ 60%)')

    # 3. BLiMP
    r = results.get('blimp', {})
    overall = r.get('overall', '?')
    print(f'\n3. BLiMP Grammaticality Judgment')
    if 'error' in r:
        print(f'   ERROR: {r["error"]}')
    else:
        print(f'   Overall : {overall}%   (chance=50%, GPT-2-large≈81%)')
        print('   Per phenomenon:')
        for k, v in r.items():
            if k != 'overall':
                print(f'     {k:<45} {v}%')

    # 4. LAMBADA
    r = results.get('lambada', {})
    print(f'\n4. LAMBADA Last-Word Prediction (n={r.get("n_examples","?")})')
    if 'error' in r:
        print(f'   ERROR: {r["error"]}')
    else:
        print(f'   Top-1 : {r.get("top1_acc","?")}%   (GPT-2-small≈45%, hard task)')
        print(f'   Top-5 : {r.get("top5_acc","?")}%')

    # 5. Embedding analogy
    r = results.get('embedding_analogy', {})
    print(f'\n5. Embedding Analogy (word2vec-style, n={r.get("total","?")})')
    if 'error' in r:
        print(f'   ERROR: {r["error"]}')
    else:
        print(f'   Accuracy : {r.get("accuracy","?")}%   '
              f'({r.get("correct","?")} / {r.get("total","?")} correct, '
              f'{r.get("skipped","?")} skipped)')

    # Timings
    print('\n' + '-' * 40)
    print('  Timings:')
    for name, t in timings.items():
        print(f'    {name:<30} {t}s')

    print('=' * 60 + '\n')


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tok(model_path: str, tok_path: str, device: torch.device):
    """Load model and tokenizer from checkpoint files."""
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from my_slm.transformer import Transformer
    from my_slm.hybrid_tokeniztion import HybridTokenizer

    # Load tokenizer
    tok = HybridTokenizer.load(tok_path)
    print(f'Tokenizer loaded  |  vocab_size={tok.vocab_size:,}')

    # Load model
    ckpt   = torch.load(model_path, map_location=device)
    cfg    = ckpt['config']
    model  = Transformer(
        vocab_size=cfg['vocab_size'],
        dim=cfg['dim'],
        depth=cfg['depth'],
        heads=cfg['heads'],
        mlp_dim=cfg['mlp_dim'],
        window=cfg['window'],
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f'Model loaded      |  {params:,} params  ({params/1e6:.1f}M)')
    return model, tok


def main():
    parser = argparse.ArgumentParser(description='Semantic evaluation for small LMs')
    parser.add_argument('--model',  required=True, help='Path to slm_checkpoint.pt')
    parser.add_argument('--tok',    required=True, help='Path to tokenizer.pkl.gz')
    parser.add_argument('--device', default='auto',
                        help='cuda / cpu / auto (default: auto)')
    parser.add_argument('--quick',  action='store_true',
                        help='Use 20%% of examples for a fast run')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    model, tok = load_model_and_tok(args.model, args.tok, device)

    report = run_all(model, tok, device, quick=args.quick)
    print_report(report)

    return report


if __name__ == '__main__':
    main()
