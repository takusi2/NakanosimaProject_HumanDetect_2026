"""
number1_selector.py

検出された複数人物の中から、参照画像との類似度(sim)が
最も高い人物を「Number 1」として選択するためのツール。

human_detector.py 側から以下のように呼び出して使用できます。

    from number1_selector import select_number1

    number1_index, number1_sim = select_number1(similarities)

similarities は、各人物の sim を入れたリスト/配列です。
例:
    [0.72, 0.91, 0.65]
    -> (1, 0.91)

人物が存在しない場合は:
    -> (-1, -1.0)
"""


def select_number1(similarities):
    """
    類似度が最も高い人物をNumber 1として選択する。

    Parameters
    ----------
    similarities : list, tuple, numpy.ndarray
        各人物の参照画像との類似度。
        配列のインデックスが人物のインデックスに対応する。

    Returns
    -------
    number1_index : int
        最も類似度が高い人物のインデックス。
        人物がいない場合は -1。

    number1_sim : float
        Number 1の類似度。
        人物がいない場合は -1.0。
    """
    if similarities is None:
        return -1, -1.0

    if len(similarities) == 0:
        return -1, -1.0

    # NaNなどを除外しながら最大値を探す
    best_index = -1
    best_sim = float("-inf")

    for i, sim in enumerate(similarities):
        try:
            sim = float(sim)
        except (TypeError, ValueError):
            continue

        if sim != sim:  # NaN
            continue

        if sim > best_sim:
            best_sim = sim
            best_index = i

    if best_index == -1:
        return -1, -1.0

    return best_index, best_sim


def is_number1(index, number1_index):
    """
    指定した人物がNumber 1か判定する。

    Parameters
    ----------
    index : int
        判定したい人物のインデックス。
    number1_index : int
        select_number1() が返したNumber 1のインデックス。

    Returns
    -------
    bool
    """
    return index == number1_index
