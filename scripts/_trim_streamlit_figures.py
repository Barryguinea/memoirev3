"""Auto-recadre les marges blanches des captures Streamlit utilisées dans le mémoire.

Lit chaque PNG dans memoire/overleaf_uqam/figures/streamlit_*.png, détecte la
boîte non-blanche, ajoute une marge fine, et sauvegarde en place.
"""

from pathlib import Path

from PIL import Image, ImageChops


def trim_white_margins(img: Image.Image, margin_px: int = 16, threshold: int = 250) -> Image.Image:
    """Retire les bandes blanches autour d'une image.

    margin_px : marge à laisser de chaque côté après recadrage
    threshold : pixel >= threshold sur les 3 canaux = considéré comme blanc
    """
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    # Étend le diff pour la robustesse
    bbox = diff.point(lambda p: 255 if p > 255 - threshold else 0).getbbox()
    if bbox is None:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - margin_px)
    top = max(0, top - margin_px)
    right = min(img.width, right + margin_px)
    bottom = min(img.height, bottom + margin_px)
    return img.crop((left, top, right, bottom))


def main() -> None:
    figures_dir = Path("memoire/overleaf_uqam/figures")
    targets = ["streamlit_individual.png", "streamlit_heatmap_compact.png"]
    for name in targets:
        path = figures_dir / name
        if not path.exists():
            print(f"  introuvable : {path}")
            continue
        img = Image.open(path)
        before = img.size
        cropped = trim_white_margins(img, margin_px=12)
        after = cropped.size
        cropped.save(path)
        print(f"  {name} : {before} -> {after}")


if __name__ == "__main__":
    main()
