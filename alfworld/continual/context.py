"""Bound generation context without dropping the system prompt or newest observation."""
import copy


def fit_context(messages, maximum, serialize, tokenizer, policy="truncate_oldest"):
    text, ids = serialize(messages, True)
    original = len(ids)
    report = {"original_input_tokens": original, "input_tokens": original,
              "dropped_messages": 0, "truncated_content_tokens": 0}
    if original <= maximum:
        return messages, text, ids, report
    if policy == "error":
        raise ValueError(f"Input has {original} tokens, exceeds {maximum}")
    prefix = 1 if messages and messages[0]["role"] == "system" else 0
    if len(messages) <= prefix:
        raise ValueError("System prompt alone exceeds the input budget")
    # Keep a valid conversation beginning at a user message; drop oldest turns first.
    candidates = [i for i in range(prefix, len(messages)) if messages[i]["role"] == "user"]
    if not candidates:
        candidates = [len(messages) - 1]
    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = messages[:prefix] + messages[candidates[mid]:]
        if len(serialize(candidate, True)[1]) <= maximum:
            hi = mid
        else:
            lo = mid + 1
    start = candidates[lo]
    kept = copy.deepcopy(messages[:prefix] + messages[start:])
    report["dropped_messages"] = start - prefix
    text, ids = serialize(kept, True)
    if len(ids) > maximum:
        # A single very long observation/curator request: remove its oldest text,
        # preserving its role and the recent suffix (including admissible commands).
        content_ids = tokenizer.encode(kept[prefix]["content"], add_special_tokens=False)
        lo, hi = 0, len(content_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            kept[prefix]["content"] = tokenizer.decode(content_ids[mid:], skip_special_tokens=False)
            if len(serialize(kept, True)[1]) <= maximum:
                hi = mid
            else:
                lo = mid + 1
        kept[prefix]["content"] = tokenizer.decode(content_ids[lo:], skip_special_tokens=False)
        report["truncated_content_tokens"] = lo
        text, ids = serialize(kept, True)
        if len(ids) > maximum:
            raise ValueError("System prompt and chat template exceed the input budget")
    report["input_tokens"] = len(ids)
    return kept, text, ids, report
