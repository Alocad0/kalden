

def is_numeric(val):
    """
    Return True if `val` is an int, a float, or a string that can be parsed as a numeric
    (int or float).  Handles scientific notation, hexadecimal integers, and the special
    floating‑point literals ``nan`` and ``inf``.
    """
    if isinstance(val, bool):
        return False

    # Direct numeric types
    if isinstance(val, (int, float)):
        return True
    
    # Strings (or bytes) that look like numbers
    if isinstance(val, (str, bytes)):
        s = val.strip()                 # remove surrounding whitespace
        if not s:                       # empty string → not numeric
            return False

        # Try integer conversion first (handles decimal, hex, octal, binary prefixes)
        try:
            int(s, 0)                   # base 0 lets Python infer the base
            return True
        except ValueError:
            pass

        # Fallback to float conversion (covers decimal floats, scientific notation,
        # and the special literals “nan”, “inf”, “-inf”, etc.)
        try:
            float(s)
            return True
        except ValueError:
            return False

    return False
