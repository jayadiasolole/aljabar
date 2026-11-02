"""
SIMULASI KONVERSI FOTO KE GRAYSCALE
===================================

Struktur program (3 layer, sama seperti simulasi rotasi 3D):

  LAYER 1 - VISUAL
      Foto asli (RGB), hasil grayscale, dan histogram tingkat keabuan.

  LAYER 2 - KONTROL
      Tombol "Upload Foto"  : memilih file foto langsung dari dalam simulasi.
      Slider "Ukuran matriks": ukuran jendela piksel yang ditampilkan (3x3, 5x5, 7x7).
      Tombol "Reset"        : kembali ke piksel tengah foto dan ukuran matriks 5x5.
      Klik pada foto        : memilih piksel yang ingin dihitung.
      Tombol panah keyboard : menggeser piksel terpilih 1 langkah (lebih presisi).

  LAYER 3 - RUMUS & MATRIKS
      Rumus grayscale, bentuk perkalian matriksnya, perhitungan piksel yang
      dipilih, dan matriks R, G, B, Gray di sekitar piksel tersebut.
      Isinya berubah mengikuti piksel yang dipilih dan ukuran matriks.

Rumus yang dipakai (ITU-R BT.601):
      Gray = 0.299*R + 0.587*G + 0.114*B

Cara menjalankan:
      pip install numpy matplotlib pillow
      python grayscale_simulasi.py
  atau langsung membuka foto:
      python grayscale_simulasi.py foto.jpg
"""

import os
import sys

import numpy as np
import matplotlib

# Tombol Upload memakai dialog file dari tkinter, jadi backend TkAgg dipilih
# supaya jendela grafik dan dialognya berjalan dalam satu sistem.
if "MPLBACKEND" not in os.environ:
    try:
        import tkinter  # noqa: F401

        matplotlib.use("TkAgg")
    except ImportError:
        print("Peringatan: tkinter tidak ditemukan, tombol Upload tidak bisa membuka dialog.")
        print("Jalankan dengan nama file: python grayscale_simulasi.py foto.jpg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# PENGATURAN
# ---------------------------------------------------------------------------
BOBOT = np.array([0.299, 0.587, 0.114])  # bobot untuk [R, G, B]
MAKS_SISI = 1200  # foto lebih besar dari ini diperkecil agar simulasi tetap lancar
UKURAN_AWAL = 5  # ukuran matriks awal (5x5)

# Warna tampilan
PERMUKAAN = "#fcfcfb"
TEKS = "#0b0b0b"
TEKS_2 = "#52514e"
GARIS_GRID = "#e4e3df"
TOMBOL = "#e9e8e3"
TOMBOL_HOVER = "#dcdbd5"
BATANG = "#6b6a66"  # batang histogram (abu-abu, sesuai tema grayscale)
AKSEN = "#2a78d6"  # penanda piksel terpilih


# ---------------------------------------------------------------------------
# FUNGSI PENGOLAHAN CITRA
# ---------------------------------------------------------------------------
def ke_rgb(img):
    """Mengubah gambar apa pun (RGBA, palet, grayscale) menjadi RGB.
    Bagian transparan diganti latar putih supaya tidak menjadi hitam."""
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        latar = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(latar, rgba).convert("RGB")
    return img.convert("RGB")


def ke_grayscale(rgb):
    """Matriks RGB (H, W, 3) -> matriks Gray (H, W).

    Operator @ adalah perkalian matriks: tiap piksel [R, G, B] dikalikan
    vektor bobot [0.299, 0.587, 0.114]. Hasilnya dibulatkan (0,5 ke atas)
    dan dibatasi ke rentang 0-255.
    """
    gray = rgb.astype(np.float64) @ BOBOT
    return np.clip(np.floor(gray + 0.5), 0, 255).astype(np.uint8)


def dialog_pilih_file(fig):
    """Membuka dialog pilih file. Mengembalikan path file, atau "" jika dibatalkan."""
    import tkinter as tk
    from tkinter import filedialog

    jendela = getattr(fig.canvas.manager, "window", None)
    pakai_jendela_sendiri = isinstance(jendela, tk.Misc)
    if pakai_jendela_sendiri:
        induk = jendela
    else:
        induk = tk.Tk()
        induk.withdraw()
    path = filedialog.askopenfilename(
        parent=induk,
        title="Pilih foto",
        filetypes=[
            ("Foto", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff *.webp"),
            ("Semua file", "*.*"),
        ],
    )
    if not pakai_jendela_sendiri:
        induk.destroy()
    return path


# ---------------------------------------------------------------------------
# SIMULASI
# ---------------------------------------------------------------------------
class SimulasiGrayscale:
    def __init__(self):
        self.rgb = None  # matriks foto asli (H, W, 3)
        self.gray = None  # matriks grayscale (H, W)
        self.hist = None  # histogram 256 nilai
        self.px = 0  # kolom piksel terpilih
        self.py = 0  # baris piksel terpilih
        self.ukuran = UKURAN_AWAL
        self.kotak = []
        self.penanda = []
        self.garis = None
        self.label_garis = None

        self._bangun_tampilan()
        self._gambar_foto()
        self._gambar_histogram()
        self._perbarui_pilihan()

    # ----- membangun jendela ------------------------------------------------
    def _bangun_tampilan(self):
        self.fig = plt.figure(figsize=(14, 8), facecolor=PERMUKAAN)
        try:
            self.fig.canvas.manager.set_window_title("Simulasi Konversi Grayscale")
            jendela = self.fig.canvas.manager.window
            if sys.platform.startswith("win"):
                jendela.state("zoomed")  # layar penuh di Windows
        except Exception:
            pass

        self.fig.text(0.04, 0.965, "Simulasi Konversi Foto ke Grayscale",
                      fontsize=16, fontweight="bold", color=TEKS, va="center")
        self.teks_info = self.fig.text(0.04, 0.938, "Belum ada foto. Klik tombol Upload Foto di bawah.",
                                       fontsize=9.5, color=TEKS_2, va="center")

        # Label tiap layer
        self.fig.text(0.04, 0.903, "LAYER 1 · VISUAL", fontsize=9, fontweight="bold", color=TEKS_2)
        self.fig.text(0.17, 0.903, "klik foto atau tekan tombol panah untuk memilih piksel",
                      fontsize=9, color=TEKS_2)
        self.fig.text(0.64, 0.903, "LAYER 3 · RUMUS & MATRIKS", fontsize=9, fontweight="bold", color=TEKS_2)
        self.fig.text(0.04, 0.098, "LAYER 2 · KONTROL", fontsize=9, fontweight="bold", color=TEKS_2)

        # Layer 1: tiga axes visual
        self.ax_asli = self.fig.add_axes([0.04, 0.53, 0.27, 0.34])
        self.ax_gray = self.fig.add_axes([0.33, 0.53, 0.27, 0.34])
        self.ax_hist = self.fig.add_axes([0.065, 0.19, 0.535, 0.27])

        # Layer 3: panel teks rumus + matriks
        self.ax_teks = self.fig.add_axes([0.64, 0.025, 0.34, 0.865])
        self.ax_teks.set_xticks([])
        self.ax_teks.set_yticks([])
        for sisi in self.ax_teks.spines.values():
            sisi.set_color(GARIS_GRID)
        self.ax_teks.set_facecolor(PERMUKAAN)
        self.teks = self.ax_teks.text(0.035, 0.975, "", transform=self.ax_teks.transAxes,
                                      va="top", ha="left", family="monospace",
                                      fontsize=8.5, color=TEKS)

        # Layer 2: kontrol
        ax_upload = self.fig.add_axes([0.04, 0.025, 0.14, 0.055])
        self.btn_upload = Button(ax_upload, "Upload Foto", color=TOMBOL, hovercolor=TOMBOL_HOVER)
        ax_reset = self.fig.add_axes([0.19, 0.025, 0.09, 0.055])
        self.btn_reset = Button(ax_reset, "Reset", color=TOMBOL, hovercolor=TOMBOL_HOVER)
        for tombol in (self.btn_upload, self.btn_reset):
            tombol.label.set_color(TEKS)
            tombol.label.set_fontsize(10)

        ax_slider = self.fig.add_axes([0.40, 0.04, 0.17, 0.03])
        self.sl_ukuran = Slider(ax_slider, "Ukuran matriks  ", 3, 7, valinit=UKURAN_AWAL,
                                valstep=2, color=AKSEN, track_color=TOMBOL_HOVER)
        self.sl_ukuran.label.set_color(TEKS)
        self.sl_ukuran.valtext.set_color(TEKS)
        self.sl_ukuran.valtext.set_x(1.07)  # geser sedikit agar tidak tertutup tuas slider
        self.sl_ukuran.valtext.set_text(f"{UKURAN_AWAL}x{UKURAN_AWAL}")

        self.btn_upload.on_clicked(self._upload)
        self.btn_reset.on_clicked(self._reset)
        self.sl_ukuran.on_changed(self._ganti_ukuran)
        self.fig.canvas.mpl_connect("button_press_event", self._klik)
        self.fig.canvas.mpl_connect("key_press_event", self._tombol_keyboard)

    # ----- menggambar -------------------------------------------------------
    def _gambar_foto(self):
        """Menggambar foto asli dan hasil grayscale (atau tulisan petunjuk)."""
        self.kotak, self.penanda = [], []
        for ax, judul in ((self.ax_asli, "Foto Asli (RGB)"), (self.ax_gray, "Hasil Grayscale")):
            ax.clear()
            ax.set_title(judul, fontsize=11, color=TEKS)
            if self.rgb is None:
                ax.set_xticks([])
                ax.set_yticks([])
                for sisi in ax.spines.values():
                    sisi.set_color(GARIS_GRID)
                ax.text(0.5, 0.5, "Belum ada foto\nklik Upload Foto", ha="center", va="center",
                        transform=ax.transAxes, color=TEKS_2, fontsize=10)

        if self.rgb is None:
            return

        self.ax_asli.imshow(self.rgb)
        self.ax_gray.imshow(self.gray, cmap="gray", vmin=0, vmax=255)  # 0 = hitam, 255 = putih
        halo = [pe.withStroke(linewidth=3.8, foreground="white")]
        for ax in (self.ax_asli, self.ax_gray):
            ax.axis("off")
            ax.set_autoscale_on(False)
            kotak = Rectangle((0, 0), 1, 1, fill=False, edgecolor=AKSEN, linewidth=1.8,
                              path_effects=halo)
            ax.add_patch(kotak)
            penanda, = ax.plot([0], [0], marker="o", markersize=11, markerfacecolor="none",
                               markeredgecolor=AKSEN, markeredgewidth=1.8, linestyle="none",
                               path_effects=halo)
            self.kotak.append(kotak)
            self.penanda.append(penanda)

    def _gambar_histogram(self):
        """Menggambar histogram: sumbu X = tingkat keabuan, sumbu Y = jumlah piksel."""
        ax = self.ax_hist
        ax.clear()
        ax.set_facecolor(PERMUKAAN)
        ax.set_xlim(-1, 256)
        ax.set_xlabel("Tingkat keabuan (0 = hitam ... 255 = putih)", color=TEKS, fontsize=9.5)
        ax.set_ylabel("Jumlah piksel", color=TEKS, fontsize=9.5)
        ax.tick_params(colors=TEKS_2, labelsize=9)
        for nama in ("top", "right"):
            ax.spines[nama].set_visible(False)
        for nama in ("left", "bottom"):
            ax.spines[nama].set_color(TEKS_2)
        ax.grid(axis="y", color=GARIS_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

        judul = "Histogram Grayscale"
        if self.hist is not None:
            judul += (f"   ·   min {int(self.gray.min())}   ·   maks {int(self.gray.max())}"
                      f"   ·   rata-rata {self.gray.mean():.1f}")
        ax.set_title(judul, loc="left", fontsize=11, color=TEKS)

        if self.hist is None:
            ax.set_yticks([])  # belum ada data, jadi sumbu Y dikosongkan
            return
        ax.bar(np.arange(256), self.hist, width=1.0, color=BATANG)
        halo = [pe.withStroke(linewidth=3.8, foreground="white")]
        self.garis = ax.axvline(0, color=AKSEN, linewidth=1.8, path_effects=halo)
        self.label_garis = ax.text(0, 0.97, "", transform=ax.get_xaxis_transform(),
                                   va="top", ha="left", fontsize=9, color=TEKS,
                                   bbox=dict(facecolor=PERMUKAAN, edgecolor="none", pad=2, alpha=0.9))

    # ----- layer 3: teks rumus & matriks --------------------------------------
    def _jendela(self):
        """Menentukan jendela n x n di sekitar piksel terpilih (digeser jika dekat tepi)."""
        tinggi, lebar = self.gray.shape
        n = min(self.ukuran, tinggi, lebar)
        x0 = int(np.clip(self.px - n // 2, 0, lebar - n))
        y0 = int(np.clip(self.py - n // 2, 0, tinggi - n))
        return x0, y0, n

    @staticmethod
    def _blok_teks(nama, matriks):
        baris = ["[" + " ".join(f"{int(v):3d}" for v in b) + "]" for b in matriks]
        return [nama.center(len(baris[0]))] + baris

    @staticmethod
    def _sisi_ke_sisi(kiri, kanan):
        return ["  " + a + "   " + b for a, b in zip(kiri, kanan)]

    def _susun_teks(self):
        bobot = "0.299 0.587 0.114"
        awal = f"  Gray = [{bobot}] x "
        pad = " " * len(awal)
        baris = [
            "RUMUS KONVERSI GRAYSCALE",
            "",
            "  Gray = 0.299 R + 0.587 G + 0.114 B",
            "",
            "Bentuk perkalian matriks:",
            pad + "[ R ]",
            awal + "[ G ]",
            pad + "[ B ]",
            "",
        ]
        if self.rgb is None:
            baris += ["Upload foto dulu. Perhitungan piksel dan",
                      "matriks di sekitarnya akan tampil di sini."]
            return "\n".join(baris)

        r, g, b = (int(v) for v in self.rgb[self.py, self.px])
        s1, s2, s3 = BOBOT[0] * r, BOBOT[1] * g, BOBOT[2] * b
        hasil = int(self.gray[self.py, self.px])
        awal2 = f"  [{bobot}] x "
        pad2 = " " * len(awal2)
        baris += [
            f"PIKSEL TERPILIH  (kolom x={self.px}, baris y={self.py})",
            "",
            f"  R = {r}, G = {g}, B = {b}",
            f"  Gray = 0.299*{r} + 0.587*{g} + 0.114*{b}",
            f"       = {s1:.3f} + {s2:.3f} + {s3:.3f}",
            f"       = {s1 + s2 + s3:.3f}  ->  {hasil}",
            "",
            "Bentuk matriks dengan angka:",
            pad2 + f"[{r:>3}]",
            awal2 + f"[{g:>3}] = [{hasil}]",
            pad2 + f"[{b:>3}]",
            "",
        ]

        x0, y0, n = self._jendela()
        potong = (slice(y0, y0 + n), slice(x0, x0 + n))
        baris += [
            f"MATRIKS PIKSEL SEKITAR ({n}x{n})",
            f"  kolom x = {x0}..{x0 + n - 1}, baris y = {y0}..{y0 + n - 1}",
            "",
        ]
        baris += self._sisi_ke_sisi(self._blok_teks("R (merah)", self.rgb[potong + (0,)]),
                                    self._blok_teks("G (hijau)", self.rgb[potong + (1,)]))
        baris.append("")
        baris += self._sisi_ke_sisi(self._blok_teks("B (biru)", self.rgb[potong + (2,)]),
                                    self._blok_teks("Gray (hasil)", self.gray[potong]))
        return "\n".join(baris)

    # ----- memperbarui tampilan ---------------------------------------------
    def _perbarui_pilihan(self):
        """Memindahkan penanda piksel, garis histogram, dan teks rumus."""
        if self.rgb is not None:
            x0, y0, n = self._jendela()
            for kotak, penanda in zip(self.kotak, self.penanda):
                kotak.set_xy((x0 - 0.5, y0 - 0.5))
                kotak.set_width(n)
                kotak.set_height(n)
                penanda.set_data([self.px], [self.py])
            nilai = int(self.gray[self.py, self.px])
            self.garis.set_xdata([nilai, nilai])
            self.label_garis.set_x(nilai)
            self.label_garis.set_ha("left" if nilai < 128 else "right")
            self.label_garis.set_text(f" piksel terpilih = {nilai} ")
        self.teks.set_text(self._susun_teks())
        self.fig.canvas.draw_idle()

    def _set_info(self, pesan):
        self.teks_info.set_text(pesan)
        self.fig.canvas.draw_idle()

    # ----- membuka foto -----------------------------------------------------
    def buka(self, path):
        """Memuat foto dari file, lalu menghitung grayscale dan histogramnya."""
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)  # luruskan foto HP yang tersimpan miring
            lebar0, tinggi0 = img.size
            img = ke_rgb(img)
            diperkecil = max(img.size) > MAKS_SISI
            if diperkecil:
                img.thumbnail((MAKS_SISI, MAKS_SISI), Image.Resampling.LANCZOS)
            rgb = np.array(img, dtype=np.uint8)
        except Exception as galat:
            self._set_info(f"Gagal membuka file: {galat}")
            return

        self.rgb = rgb
        self.gray = ke_grayscale(rgb)
        self.hist = np.bincount(self.gray.ravel(), minlength=256)
        tinggi, lebar = self.gray.shape
        self.px, self.py = lebar // 2, tinggi // 2

        info = f"{os.path.basename(path)}  -  {lebar} x {tinggi} piksel"
        if diperkecil:
            info += f"  (diperkecil dari {lebar0} x {tinggi0} agar lancar)"
        self.teks_info.set_text(info)

        self._gambar_foto()
        self._gambar_histogram()
        self._perbarui_pilihan()

    # ----- event kontrol ----------------------------------------------------
    def _upload(self, _event):
        try:
            path = dialog_pilih_file(self.fig)
        except Exception as galat:
            self._set_info(f"Dialog upload tidak tersedia ({galat}). "
                           "Jalankan: python grayscale_simulasi.py foto.jpg")
            return
        if path:
            self.buka(path)

    def _reset(self, _event):
        self.sl_ukuran.set_val(UKURAN_AWAL)  # memicu _ganti_ukuran
        if self.rgb is not None:
            tinggi, lebar = self.gray.shape
            self.px, self.py = lebar // 2, tinggi // 2
            self._perbarui_pilihan()

    def _ganti_ukuran(self, nilai):
        self.ukuran = int(nilai)
        self.sl_ukuran.valtext.set_text(f"{self.ukuran}x{self.ukuran}")
        self._perbarui_pilihan()

    def _klik(self, event):
        if self.rgb is None or event.inaxes not in (self.ax_asli, self.ax_gray):
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # sedang memakai zoom/pan, jangan ubah pilihan
        tinggi, lebar = self.gray.shape
        self.px = int(np.clip(round(event.xdata), 0, lebar - 1))
        self.py = int(np.clip(round(event.ydata), 0, tinggi - 1))
        self._perbarui_pilihan()

    def _tombol_keyboard(self, event):
        if self.rgb is None:
            return
        geser = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
        if event.key not in geser:
            return
        dx, dy = geser[event.key]
        tinggi, lebar = self.gray.shape
        self.px = int(np.clip(self.px + dx, 0, lebar - 1))
        self.py = int(np.clip(self.py + dy, 0, tinggi - 1))
        self._perbarui_pilihan()


def main():
    simulasi = SimulasiGrayscale()
    if len(sys.argv) > 1:
        simulasi.buka(sys.argv[1])
    plt.show()


if __name__ == "__main__":
    main()
