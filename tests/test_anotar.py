"""Testa a lógica de salvar do anotador, sem abrir janela (tkinter é trocado por versões falsas)."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def anotar(monkeypatch):
    for nome in ("tkinter", "tkinter.messagebox", "tkinter.ttk"):
        monkeypatch.setitem(sys.modules, nome, MagicMock())
    monkeypatch.setitem(sys.modules, "PIL.ImageTk", MagicMock())
    sys.modules.pop("anotar", None)
    import anotar as modulo
    yield modulo
    sys.modules.pop("anotar", None)


def make_app(anotar, tmp_path):
    """Monta o objeto sem chamar o __init__ (que abriria a janela)."""
    app = anotar.Application.__new__(anotar.Application)
    app.labels_dir = tmp_path
    app.current_path = Path("foto1.png")
    app.original_w, app.original_h = 1000, 500
    app.classes = ["gato", "cachorro"]
    app.index = 0
    app.images = [Path("foto1.png")]
    app.annotations = []
    app._show_current = lambda: None
    return app


def test_confirmar_converte_para_tamanho_original(anotar, tmp_path):
    app = make_app(anotar, tmp_path)
    # caixa de (100,100) a (300,200) no canvas 500x500: centro em 40% e 30%, tamanho 40% x 20%
    app.annotations = [{"box": (100, 100, 300, 200), "class": "cachorro", "ids": []}]
    anotar.messagebox = MagicMock()

    app.confirm()

    linha = (tmp_path / "foto1.txt").read_text(encoding="utf-8").strip()
    assert linha == "1 0.400000 0.300000 0.400000 0.200000"
    assert app.index == 1


def test_confirmar_sem_caixas_nao_salva(anotar, tmp_path):
    app = make_app(anotar, tmp_path)
    anotar.messagebox = MagicMock()
    app.confirm()
    assert not list(tmp_path.glob("*.txt"))
    anotar.messagebox.showwarning.assert_called_once()


def test_esc_cria_anotacao_vazia(anotar, tmp_path):
    app = make_app(anotar, tmp_path)
    app.skip()
    assert (tmp_path / "foto1.txt").read_text(encoding="utf-8") == ""
    assert app.index == 1
