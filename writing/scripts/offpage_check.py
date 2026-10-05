"""Report words placed outside the page box of a PDF (text lost to the reader)."""
import sys
import pymupdf


def offpage(path):
    res = []
    for i, p in enumerate(pymupdf.open(path)):
        H, W = p.rect.height, p.rect.width
        ws = p.get_text("words", clip=pymupdf.INFINITE_RECT(), flags=0)
        out = [w for w in ws if w[3] > H + 1 or w[2] > W + 1 or w[1] < -1 or w[0] < -1]
        if out:
            res.append((i + 1, len(out), round(max(w[3] for w in out))))
    return res


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, offpage(f) or "no off-page text")
