import tkinter as tk
from PixelArtConverter import PixelArtConverter


class Application(tk.Tk):
    def __init__(self, converter, pixel_size=10):
        super().__init__()
        self.converter = converter
        self.pixel_size = pixel_size
        self.canvas_left = tk.Canvas(self, width=28 * pixel_size, height=28 * pixel_size)
        self.canvas_left.grid(row=0, column=0)
        self.canvas_right = tk.Canvas(self, width=28 * pixel_size, height=28 * pixel_size)
        self.canvas_right.grid(row=0, column=1)
        self.update_canvases()
        self.after(1000, self.update)

    def update_canvases(self):
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        for i in range(28):
            for j in range(28):
                color = 'black' if self.converter.output_image[i][j] else 'white'
                x1 = j * self.pixel_size
                y1 = i * self.pixel_size
                self.canvas_left.create_rectangle(x1, y1, x1 + self.pixel_size, y1 + self.pixel_size, fill=color,
                                                  outline='')
                color = 'black' if self.converter.input_image[i][j] else 'white'
                self.canvas_right.create_rectangle(x1, y1, x1 + self.pixel_size, y1 + self.pixel_size, fill=color,
                                                   outline='')

    def update(self):
        self.converter.next_step()
        self.update_canvases()
        self.after(1000, self.update)


# Пример целевого изображения (диагональ)
target_image = [[False] * 28 for _ in range(28)]
for i in range(28):
    target_image[i][i] = True

converter = PixelArtConverter(target_image)
app = Application(converter)
app.mainloop()
