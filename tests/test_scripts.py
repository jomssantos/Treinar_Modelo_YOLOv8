import verificar_anotacoes
import treinar
import testar_modelo
from yolo_utils import save_data_yaml


def test_treinar_passa_os_argumentos_ao_yolo(make_dataset, fake_ultralytics, tmp_path, capsys):
    root = make_dataset()
    data = tmp_path / "data.yaml"
    save_data_yaml(data, ["objeto"], root)

    code = treinar.main(["--data", str(data), "--dataset", str(root), "--epocas", "5", "--imgsz", "416", "--nome", "teste", "--device", "cpu"])

    assert code == 0
    assert fake_ultralytics["model"] == "yolov8n.pt"
    assert fake_ultralytics["train"] == dict(data=str(data), epochs=5, imgsz=416, batch=16, name="teste", device="cpu")
    assert "best.pt" in capsys.readouterr().out


def test_treinar_nao_passa_device_quando_nao_informado(make_dataset, fake_ultralytics, tmp_path):
    root = make_dataset()
    data = tmp_path / "data.yaml"
    save_data_yaml(data, ["objeto"], root)
    treinar.main(["--data", str(data), "--dataset", str(root)])
    assert "device" not in fake_ultralytics["train"]


def test_treinar_bloqueia_dataset_com_problema(make_dataset, fake_ultralytics, tmp_path, capsys):
    root = make_dataset()
    (root / "labels" / "train" / "img0.txt").write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    save_data_yaml(data, ["objeto"], root)

    code = treinar.main(["--data", str(data), "--dataset", str(root)])

    assert code == 1
    assert "train" not in fake_ultralytics          # o treino nem começou
    assert "fora do intervalo" in capsys.readouterr().out


def test_treinar_pode_pular_a_verificacao(make_dataset, fake_ultralytics, tmp_path):
    root = make_dataset(phases=("train",))              # sem val: normalmente bloqueia
    data = tmp_path / "data.yaml"
    save_data_yaml(data, ["objeto"], root)
    assert treinar.main(["--data", str(data), "--dataset", str(root), "--pular-verificacao"]) == 0


def test_testar_modelo_lista_deteccoes(fake_ultralytics, capsys):
    code = testar_modelo.main(["--imagem", "foto.jpg", "--conf", "0.3"])
    saida = capsys.readouterr().out

    assert code == 0
    assert fake_ultralytics["predict"]["conf"] == 0.3 and fake_ultralytics["predict"]["save"] is True
    assert "2 detecção(ões)" in saida and "gato 0.91" in saida and "cachorro 0.55" in saida


def test_verificar_anotacoes_relatorio_e_imagens_com_caixas(make_dataset, tmp_path, capsys):
    root = make_dataset()
    data = tmp_path / "data.yaml"
    save_data_yaml(data, ["objeto"], root)
    saida = tmp_path / "conferencia"

    code = verificar_anotacoes.main(["--dataset", str(root), "--data-yaml", str(data), "--salvar", str(saida)])

    assert code == 0
    assert "Dataset pronto para treinar" in capsys.readouterr().out
    assert sorted(p.name for p in saida.iterdir()) == ["img0.png", "img1.png"]


def test_verificar_anotacoes_retorna_erro_quando_ha_problemas(make_dataset, tmp_path):
    root = make_dataset(phases=("train",))
    data = tmp_path / "data.yaml"
    save_data_yaml(data, ["objeto"], root)
    assert verificar_anotacoes.main(["--dataset", str(root), "--data-yaml", str(data)]) == 1
