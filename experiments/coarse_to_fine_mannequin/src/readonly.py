"""初期化後に値を書き換えられない設定オブジェクトの共通処理。"""


class ReadOnlySettings:
    """``_lock_settings()`` の後は、通常の代入による変更を拒否する。"""

    def _lock_settings(self) -> None:
        """全設定値の代入が終わった時点で呼び出す。"""
        object.__setattr__(self, "_settings_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_settings_locked", False):
            raise AttributeError("設定値は初期化後に変更できません")
        object.__setattr__(self, name, value)
