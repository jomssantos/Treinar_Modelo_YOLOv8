import pytest

from yolo_utils import (
    load_classes, parse_label_line, pending_images, save_data_yaml, to_yolo_line, validate_dataset,
)


def test_to_yolo_line_normaliza_e_ordena_os_cantos():
    # caixa de (50,20) a (150,60) em uma imagem 200x100, desenhada "de trás para frente"
    line = to_yolo_line(2, (150, 60), (50, 20), 200, 100)
    assert line == "2 0.500000 0.400000 0.500000 0.400000"


def test_to_yolo_line_limita_ao_tamanho_da_imagem():
    line = to_yolo_line(0, (-10, -10), (500, 500), 200, 100)
    assert line == "0 0.500000 0.500000 1.000000 1.000000"


def test_to_yolo_line_descarta_caixa_minuscula():
    assert to_yolo_line(0, (10, 10), (11, 11), 200, 100) is None


def test_classes_ida_e_volta(tmp_path):
    arquivo = tmp_path / "data.yaml"
    save_data_yaml(arquivo, ["gato", "cachorro"], "dataset")
    assert load_classes(arquivo) == ["gato", "cachorro"]
    texto = arquivo.read_text(encoding="utf-8")
    assert "nc: 2" in texto and "dataset/images/train" in texto


def test_classes_aceita_formato_de_dicionario(tmp_path):
    arquivo = tmp_path / "data.yaml"
    arquivo.write_text("names:\n  0: a\n  1: b\n", encoding="utf-8")
    assert load_classes(arquivo) == ["a", "b"]
    assert load_classes(tmp_path / "nao-existe.yaml") == []


def test_parse_label_line_valida_valores():
    assert parse_label_line("1 0.5 0.5 0.2 0.2", num_classes=2)[0] == 1
    for ruim in ["1 0.5 0.5", "x 0.5 0.5 0.2 0.2", "3 0.5 0.5 0.2 0.2", "0 1.5 0.5 0.2 0.2", "0 0.5 0.5 0 0.2"]:
        with pytest.raises(ValueError):
            parse_label_line(ruim, num_classes=2)


def test_pending_images_ignora_as_ja_anotadas_e_nao_imagens(make_dataset):
    root = make_dataset(phases=("train",), per_phase=3)
    (root / "labels" / "train" / "img1.txt").unlink()
    (root / "images" / "train" / "notas.txt").write_text("não é imagem", encoding="utf-8")
    nomes = [p.name for p in pending_images(root / "images" / "train", root / "labels" / "train")]
    assert nomes == ["img1.png"]


def test_dataset_valido(make_dataset):
    report = validate_dataset(make_dataset(), num_classes=1)
    assert report.ok and report.errors == []
    assert report.class_counts[0] == 4


def test_dataset_aponta_problemas(make_dataset):
    root = make_dataset()
    (root / "labels" / "train" / "img0.txt").write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "labels" / "val" / "fantasma.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "labels" / "val" / "img1.txt").unlink()
    report = validate_dataset(root, num_classes=1)
    texto = "\n".join(report.errors)
    assert not report.ok
    assert "classe 5 fora do intervalo" in texto
    assert "fantasma.txt" in texto
    assert any("sem anotação" in w for w in report.warnings)


def test_anotacao_vazia_conta_como_imagem_sem_objetos(make_dataset):
    root = make_dataset()
    (root / "labels" / "train" / "img0.txt").write_text("", encoding="utf-8")
    report = validate_dataset(root, num_classes=1)
    assert report.ok and report.empty_labels["train"] == 1


def test_fase_vazia_e_erro(make_dataset):
    report = validate_dataset(make_dataset(phases=("train",)), num_classes=1)
    assert any("[val] nenhuma imagem" in e for e in report.errors)
