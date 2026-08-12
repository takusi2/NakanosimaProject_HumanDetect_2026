# tools/aspect_filter.py

def is_valid_aspect_ratio(bw: int, bh: int, ratio_min: float, ratio_max: float) -> bool:
    """
    バウンディングボックスの縦横比（height / width）が
    指定範囲内かどうかを判定する。

    Args:
        bw: バウンディングボックスの幅
        bh: バウンディングボックスの高さ
        ratio_min: 縦横比の最小値（この値未満はFalse）
        ratio_max: 縦横比の最大値（この値超過はFalse）

    Returns:
        True: 縦横比が範囲内（人型として有効）
        False: 縦横比が範囲外（横長すぎ・正方形・縦長すぎ）
    """
    if bw <= 0:
        return False

    aspect_ratio = bh / bw

    return ratio_min <= aspect_ratio <= ratio_max