# Treinar_Modelo_YOLOv8

Passo a passo em Python para treinar o meu próprio detector de objetos com o [YOLOv8](https://docs.ultralytics.com). Eu junto tudo num lugar só: anoto as imagens numa ferramenta gráfica simples, confiro o dataset, treino e testo em uma foto ou na webcam.

Comecei com uma única classe, mas funciona com quantas você quiser.

## Fluxo

```text
anotar.py -> verificar_anotacoes.py -> treinar.py -> testar_modelo.py / testar_webcam.py
```

Para cada imagem, o anotador cria um `.txt` com o mesmo nome. Cada linha é um objeto no formato do YOLO:

```text
classe  centro_x  centro_y  largura  altura
```

Os quatro últimos valores ficam entre 0 e 1, relativos ao tamanho da imagem. O `anotar.py` faz essa conversão sozinho.

## Estrutura

```text
dataset/
  images/train, images/val    imagens
  labels/train, labels/val    anotações (.txt)
anotar.py                 anotação gráfica
verificar_anotacoes.py    relatório do dataset e caixas desenhadas
treinar.py                treino
testar_modelo.py          teste em uma imagem
testar_webcam.py          teste em tempo real
yolo_utils.py             formato YOLO, classes e validação
data.yaml                 classes e caminhos
tests/
```

Imagens, anotações, `runs/` e arquivos `.pt`/`.onnx` ficam fora do Git de propósito, para o repositório continuar leve e sem dados pessoais.

## Instalação

Precisa de Python 3.10+.

```bash
git clone https://github.com/SEU-USUARIO/Treinar_Modelo_YOLOv8.git
cd Treinar_Modelo_YOLOv8

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

O anotador usa Tkinter. No Windows e no macOS já vem com o Python; no Linux: `sudo apt install python3-tk`. Com placa NVIDIA, instale o PyTorch com CUDA seguindo [pytorch.org](https://pytorch.org/get-started/locally/). Sem GPU funciona, só demora mais.

## Passo 1: imagens

Coloque as imagens (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) em duas pastas:

- `dataset/images/train`: cerca de 80%.
- `dataset/images/val`: cerca de 20%, que o modelo nunca vê durante o aprendizado.

Misture bem. Se a validação for parecida demais com o treino (quadros seguidos do mesmo vídeo, por exemplo), a nota do modelo parece melhor do que é.

## Passo 2: anotar

```bash
python anotar.py
```

1. Digite o nome da classe e aperte **Enter**.
2. Clique na classe na lista.
3. Arraste o mouse sobre o objeto para desenhar a caixa. Repita para cada objeto.
4. **Voltar** apaga a última caixa; **Confirmar** salva e vai para a próxima.
5. Imagem sem objeto: **ESC**. Ela vira exemplo de fundo e ajuda o modelo a não inventar detecção.

Dá para fechar e voltar depois, as imagens já anotadas não reaparecem. As classes ficam no `data.yaml`, então os IDs se mantêm. Para começar pela validação: `python anotar.py --fase val`. A imagem aparece esticada em 500x500, mas as caixas são convertidas para o tamanho real ao salvar.

## Passo 3: conferir

```bash
python verificar_anotacoes.py
python verificar_anotacoes.py --ver train            # abre as imagens com as caixas
python verificar_anotacoes.py --salvar conferencia   # grava numa pasta
```

O relatório mostra imagens por fase, caixas por classe e aponta classe fora do `data.yaml`, coordenada fora de 0 a 1, anotação sem imagem, imagem sem anotação e fase vazia.

## Passo 4: treinar

```bash
python treinar.py
python treinar.py --nome meu_detector --epocas 100 --imgsz 416
python treinar.py --device cpu       # ou --device 0 para a GPU
python treinar.py --batch 8          # se faltar memória
```

O `treinar.py` roda a mesma verificação antes e se recusa a treinar se achar erro. O melhor modelo fica em `runs/detect/modelo/weights/best.pt`, junto com os gráficos de desempenho. Treinando duas vezes com o mesmo `--nome`, o YOLO cria `modelo2`, `modelo3`... e você aponta a pasta certa com `--modelo` nos testes.

| Opção | Padrão | Para que serve |
| --- | --- | --- |
| `--modelo` | `yolov8n.pt` | modelo base; `yolov8s.pt` ou maior dá mais precisão e mais lentidão |
| `--epocas` | `80` | voltas pelo dataset inteiro |
| `--imgsz` | `320` | tamanho no treino; maior pega objetos pequenos, mas é mais lento |
| `--batch` | `16` | imagens por passo |
| `--device` | automático | `cpu` ou número da GPU |
| `--nome` | `modelo` | pasta de resultados |
| `--pular-verificacao` | desligado | treina mesmo com problemas |

## Passo 5: testar

```bash
python testar_modelo.py --imagem foto.jpg
python testar_modelo.py --imagem foto.jpg --conf 0.3 --mostrar
python testar_webcam.py --conf 0.4
```

O `testar_modelo.py` lista cada detecção com classe e confiança e salva a imagem em `runs/detect/predict`. Na webcam, **C** salva o quadro em `foto.jpg` (sem caixas) e **ESC** ou **Q** sai.

`--conf` é a confiança mínima (padrão `0.5`): baixe para detectar mais, suba para errar menos. `--camera` escolhe a câmera (padrão `0`).

## Dicas

- **Variedade importa mais que quantidade.** Fotografe com luz, fundo, ângulo e distância diferentes, do jeito que o objeto vai aparecer na prática. Para uma classe simples, comece com algumas centenas de imagens.
- **Inclua imagens sem o objeto** (ESC no anotador).
- **Seja consistente nas caixas:** decida se pega o objeto inteiro ou só uma parte e mantenha o critério.
- **Olhe os gráficos em `runs/detect/<nome>`.** Se a perda de validação sobe enquanto a de treino desce, o modelo está decorando: junte mais imagens ou reduza as épocas.
- **Teste em imagens novas**, que não estavam no treino nem na validação.

## Problemas comuns

- **`No module named 'tkinter'`:** no Linux, `sudo apt install python3-tk`.
- **Treino não começa e lista erros:** rode `verificar_anotacoes.py`. Os casos mais comuns são classe fora do `data.yaml` e pasta `train` ou `val` vazia.
- **`CUDA out of memory`:** reduza `--batch` (8 ou 4) ou `--imgsz`.
- **Treino lento:** sem GPU é esperado. Teste a ideia com poucas épocas e `--imgsz` menor antes do treino completo.
- **Não detecta nada:** baixe o `--conf` (0.25), confira as caixas com `--ver train` e junte mais imagens.
- **Detecta onde não existe:** suba o `--conf` e inclua mais imagens de fundo.
- **`best.pt` não encontrado:** veja o nome da pasta em `runs/detect/` e passe o caminho em `--modelo`.
- **Câmera não abre:** tente `--camera 1` e veja se outro programa está usando a webcam.

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Não treinam nada nem abrem janela: o `ultralytics` e o Tkinter são trocados por versões falsas.

## Licença

MIT. Veja [LICENSE](LICENSE).
