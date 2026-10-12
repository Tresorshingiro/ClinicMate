import ast, json, math, re, unicodedata

LETTERS = "ABCDE"

def is_missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

def clean_text(x):
    if is_missing(x):
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    s = s.replace("�", "-")                  # broken bullet characters (�) in some SAQs
    s = s.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_qtype(x):
    s = str(x).lower()
    if "mcq" in s:
        return "mcq"
    if "saq" in s:
        return "saq"
    if "consumer" in s:
        return "cq"
    return "unknown"

def parse_options(raw):
    if is_missing(raw):
        return []
    d = raw
    if not isinstance(raw, (dict, list)):
        s = str(raw).strip()
        if not s:
            return []
        try:
            d = json.loads(s)
        except Exception:
            try:
                d = ast.literal_eval(s)
            except Exception:
                return []
    if isinstance(d, list):
        d = {f"option{i + 1}": v for i, v in enumerate(d)}
    items = []
    for k, v in d.items():
        text = clean_text(v)
        if text and text.lower() not in {"n/a", "na", "none", "null"}:
            items.append((re.sub(r"\s+", "", str(k).lower()), text))
    items.sort(key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0))
    return items[: len(LETTERS)]

def parse_correct(raw, options):
    if is_missing(raw) or not options:
        return []
    keys = [k for k, _ in options]
    s = str(raw).strip()
    letters = []
    nums = re.findall(r"option\s*(\d+)", s, flags=re.I)
    if nums:
        for n in nums:
            k = f"option{n}"
            if k in keys:
                letters.append(LETTERS[keys.index(k)])
    elif re.fullmatch(r"[A-Ea-e]", s):
        idx = ord(s.upper()) - ord("A")
        if idx < len(options):
            letters.append(s.upper())
    else:
        for i, (_, t) in enumerate(options):
            if t.lower() == s.lower():
                letters.append(LETTERS[i])
    return sorted(set(letters))


def dedup_key(question, context=""):
    return re.sub(r"[^a-z0-9]", "", f"{context} {question}".lower())