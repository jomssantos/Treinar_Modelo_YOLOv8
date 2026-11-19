# uso: python anotar.py [--fase val]  |  ESC = imagem sem objetos
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from yolo_utils import load_classes, pending_images, save_data_yaml, to_yolo_line

CANVAS_SIZE = 500
PLACEHOLDER = "Digite o nome da classe..."
COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#5371e0", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#0044FF"]


class Application:
    def __init__(self, dataset_dir: str = "dataset", data_yaml: str = "data.yaml", phase: str = "train") -> None:
        self.dataset = Path(dataset_dir)
        self.data_yaml = Path(data_yaml)
        self.classes: list[str] = load_classes(self.data_yaml)
        self.current_class: str | None = None
        self.images: list[Path] = []
        self.index = 0
        self.annotations: list[dict] = []  # caixas da imagem atual
        self.drawing = False
        self.start_point = (0, 0)
        self.in_placeholder = True
        self.photo = None

        self.root = tk.Tk()
        self.root.title("Anotador")
        self.root.configure(bg="#222A31")
        self.root.geometry("720x550")
        self.root.resizable(False, False)

        self._build_widgets()
        self._load_phase(phase)
        self.root.after(0, self._show_current)
        self.root.mainloop()

    def _build_widgets(self) -> None:
        panel = tk.Frame(self.root, bg="#2C3E50")
        panel.place(relx=0.01, rely=0.01, relwidth=0.25, relheight=0.9)

        frame = tk.Frame(self.root, bg="#95A5A6")
        frame.place(relx=0.28, rely=0.01, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.canvas = tk.Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE, cursor="cross")
        self.canvas.place(relx=0, rely=0, relheight=1, relwidth=1)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.root.bind("<Escape>", self.skip)

        self.entry = tk.Entry(panel, fg="gray")
        self.entry.insert(0, PLACEHOLDER)
        self.entry.bind("<FocusIn>", self.on_entry_focus)
        self.entry.bind("<FocusOut>", self.on_entry_blur)
        self.entry.bind("<Return>", self.on_entry_enter)
        self.entry.place(relx=0.01, rely=0.05, relwidth=0.95, relheight=0.06)

        self.listbox = ttk.Treeview(panel, columns=("classes",), show="headings")
        self.listbox.heading("classes", text="Classes")
        self.listbox.column("classes", width=120)
        self.listbox.place(relx=0.01, rely=0.12, relwidth=0.75, relheight=0.3)
        self.listbox.bind("<ButtonRelease-1>", self.on_class_click)
        for name in self.classes:
            self.listbox.insert("", "end", values=(name,))

        self.info = tk.Label(panel, text="", bg="#2C3E50", fg="white", justify="left", anchor="nw")
        self.info.place(relx=0.01, rely=0.45, relwidth=0.95, relheight=0.2)

        self.btn_undo = tk.Button(panel, text="Voltar", command=self.undo, bg="#a1a1a1")
        self.btn_confirm = tk.Button(panel, text="Confirmar", command=self.confirm, bg="#4b84ff")
        self.btn_confirm.place(relx=0.01, rely=0.8, relwidth=0.95, relheight=0.06)

    def _load_phase(self, phase: str) -> None:
        self.phase = phase
        self.images_dir = self.dataset / "images" / phase
        self.labels_dir = self.dataset / "labels" / phase
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.images = pending_images(self.images_dir, self.labels_dir)
        self.index = 0

    def _show_current(self) -> None:
        while self.index < len(self.images):
            path = self.images[self.index]
            image = cv2.imread(str(path))
            if image is not None:
                break
            print(f"Não abriu {path.name}, pulando")
            self.index += 1
        else:
            self._finish_phase()
            return

        self.current_path = path
        self.original_h, self.original_w = image.shape[:2]
        self.annotations.clear()
        self.btn_undo.place_forget()

        # exibe em 500x500, converte de volta ao salvar
        resized = cv2.resize(image, (CANVAS_SIZE, CANVAS_SIZE))
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.info.config(text=f"Fase: {self.phase}\nImagem {self.index + 1} de {len(self.images)}\n{path.name}")

    def _finish_phase(self) -> None:
        if self.phase == "train" and messagebox.askyesno("Fim do treino", "Anotar as imagens de validação?"):
            self._load_phase("val")
            self._show_current()
            return
        self.root.destroy()

    def color_of(self, name: str) -> str:
        return COLORS[self.classes.index(name) % len(COLORS)]

    def on_entry_focus(self, _event) -> None:
        if self.in_placeholder:
            self.in_placeholder = False
            self.entry.delete(0, tk.END)
            self.entry.config(fg="black")

    def on_entry_blur(self, _event) -> None:
        if not self.entry.get():
            self._reset_entry()

    def _reset_entry(self) -> None:
        self.entry.delete(0, tk.END)
        self.entry.insert(0, PLACEHOLDER)
        self.entry.config(fg="gray")
        self.in_placeholder = True

    def on_entry_enter(self, _event) -> None:
        text = self.entry.get().strip().lower()
        if text and not self.in_placeholder and text not in self.classes:
            self.classes.append(text)
            save_data_yaml(self.data_yaml, self.classes, self.dataset)
            self.listbox.insert("", "end", values=(text,))
        self._reset_entry()
        self.root.focus_set()

    def on_class_click(self, _event) -> None:
        item = self.listbox.focus()
        if item:
            self.current_class = str(self.listbox.item(item, "values")[0])

    @staticmethod
    def _clamp(value: int) -> int:
        return max(0, min(CANVAS_SIZE, value))

    def on_mouse_down(self, event) -> None:
        if not self.current_class:
            self.info.config(text="Escolha uma classe\nna lista antes.")
            return
        self.start_point = (self._clamp(event.x), self._clamp(event.y))
        self.drawing = True

    def on_mouse_move(self, event) -> None:
        if not self.drawing:
            return
        self.canvas.delete("temp")
        self.canvas.create_rectangle(
            *self.start_point, self._clamp(event.x), self._clamp(event.y),
            outline=self.color_of(self.current_class), width=2, tags="temp",
        )

    def on_mouse_up(self, event) -> None:
        if not self.drawing:
            return
        self.drawing = False
        self.canvas.delete("temp")

        x0, y0 = self.start_point
        x1, y1 = self._clamp(event.x), self._clamp(event.y)
        if abs(x1 - x0) < 3 or abs(y1 - y0) < 3:
            return  # clique sem arrastar

        color = self.color_of(self.current_class)
        rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2)
        label = self.canvas.create_text(min(x0, x1), min(y0, y1) - 14, text=self.current_class, anchor="nw", fill=color)
        self.annotations.append({"box": (x0, y0, x1, y1), "class": self.current_class, "ids": [rect, label]})
        self.btn_undo.place(relx=0.01, rely=0.72, relwidth=0.95, relheight=0.06)

    def undo(self) -> None:
        if self.annotations:
            for item_id in self.annotations.pop()["ids"]:
                self.canvas.delete(item_id)
        if not self.annotations:
            self.btn_undo.place_forget()

    def _label_path(self) -> Path:
        return self.labels_dir / f"{self.current_path.stem}.txt"

    def confirm(self) -> None:
        if not self.annotations:
            messagebox.showwarning("Sem objetos", "Marque ao menos um objeto. Sem nenhum, aperte ESC.")
            return

        sx, sy = self.original_w / CANVAS_SIZE, self.original_h / CANVAS_SIZE
        lines = []
        for ann in self.annotations:
            x0, y0, x1, y1 = ann["box"]
            line = to_yolo_line(
                self.classes.index(ann["class"]), (x0 * sx, y0 * sy), (x1 * sx, y1 * sy), self.original_w, self.original_h
            )
            if line:
                lines.append(line)

        self._label_path().write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Salvo: {self._label_path()} ({len(lines)} caixa(s))")
        self.index += 1
        self._show_current()

    def skip(self, _event=None) -> None:
        # anotação vazia vira exemplo de fundo
        self._label_path().write_text("", encoding="utf-8")
        print(f"Sem objetos: {self.current_path.name}")
        self.index += 1
        self._show_current()


def main() -> None:
    parser = argparse.ArgumentParser(description="Anotador de imagens no formato YOLO")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--data-yaml", default="data.yaml")
    parser.add_argument("--fase", choices=["train", "val"], default="train")
    args = parser.parse_args()
    Application(args.dataset, args.data_yaml, args.fase)


if __name__ == "__main__":
    main()
