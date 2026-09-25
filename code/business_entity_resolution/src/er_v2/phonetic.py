"""Small, language-agnostic consonant classes; never a hard matching rule."""
import polars as pl


def phonetic_expr(token: pl.Expr) -> pl.Expr:
    value = token.str.to_lowercase().str.replace_all("[^a-z]", "")
    for letters, code in (("bp", "1"), ("dt", "2"), ("gkqc", "3"),
                          ("szxj", "4"), ("vwf", "5"), ("mn", "6"),
                          ("l", "7"), ("r", "8")):
        value = value.str.replace_all(f"[{letters}]", code)
    value = value.str.replace_all("[a-z]", "")
    for code in "12345678":
        value = value.str.replace_all(code + "+", code)
    return value
