import hashlib
import random


class PixelArtConverter:
    def __init__(self, target_image):
        self.target_image = target_image  # 28x28 матрица булевых значений (True - черный)
        self.inverted_target = [[not pixel for pixel in row] for row in target_image]
        self.input_image = [[False] * 28 for _ in range(28)]  # Изначально белое
        self.output_image = [[True] * 28 for _ in range(28)]  # Изначально черное
        self.phase = 'filling'  # 'filling' или 'clearing'

    def hash_input_image(self):
        byte_array = bytearray()
        for row in self.input_image:
            for pixel in row:
                byte_array.append(1 if pixel else 0)
        return hashlib.sha256(byte_array).digest()

    def update_output_image(self):
        if all(all(row) for row in self.input_image):
            self.output_image = [row.copy() for row in self.target_image]
        elif all(not pixel for row in self.input_image for pixel in row):
            self.output_image = [row.copy() for row in self.inverted_target]
        else:
            hash_bytes = self.hash_input_image()
            bits = []
            for byte in hash_bytes:
                bits.extend([(byte >> i) & 1 for i in range(7, -1, -1)])
            bits = (bits * (784 // len(bits) + 1))[:784]
            index = 0
            for i in range(28):
                for j in range(28):
                    self.output_image[i][j] = bits[index]
                    index += 1

    def next_step(self):
        if self.phase == 'filling':
            white_pixels = [(i, j) for i in range(28) for j in range(28) if not self.input_image[i][j]]
            if not white_pixels:
                self.phase = 'clearing'
                return
            hash_bytes = self.hash_input_image()
            random.seed(int.from_bytes(hash_bytes[:4], 'big'))
            selected = random.sample(white_pixels, min(3, len(white_pixels)))
            for i, j in selected:
                self.input_image[i][j] = True
        else:
            black_pixels = [(i, j) for i in range(28) for j in range(28) if self.input_image[i][j]]
            if not black_pixels:
                self.phase = 'filling'
                return
            hash_bytes = self.hash_input_image()
            random.seed(int.from_bytes(hash_bytes[:4], 'big'))
            selected = random.sample(black_pixels, min(3, len(black_pixels)))
            for i, j in selected:
                self.input_image[i][j] = False
        self.update_output_image()
