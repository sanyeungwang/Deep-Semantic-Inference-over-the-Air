#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import fitz  # PyMuPDF


def merge_2x2(top_left, top_right, bottom_left, bottom_right, output_pdf):
    docs = [fitz.open(p) for p in [top_left, top_right, bottom_left, bottom_right]]
    pages = [d[0] for d in docs]
    sizes = [(pg.rect.width, pg.rect.height) for pg in pages]

    # column width = maximum width of the two pages above and below the column;
    # row height = maximum height of the two pages to the left and right of the row (without scaling).
    w_left = max(sizes[0][0], sizes[2][0])
    w_right = max(sizes[1][0], sizes[3][0])
    h_top = max(sizes[0][1], sizes[1][1])
    h_bot = max(sizes[2][1], sizes[3][1])

    out_w = w_left + w_right
    out_h = h_top + h_bot

    out = fitz.open()
    page = out.new_page(width=out_w, height=out_h)

    def place(doc, idx, x, y):  # top left alignment
        w = doc[idx].rect.width
        h = doc[idx].rect.height
        page.show_pdf_page(fitz.Rect(x, y, x + w, y + h), doc, idx)

    # (a) top left, (b) top right, (c) bottom left, (d) bottom right
    place(docs[0], 0, 0, 0)
    place(docs[1], 0, w_left, 0)
    place(docs[2], 0, 0, h_top)
    place(docs[3], 0, w_left, h_top)

    out.save(output_pdf)
    out.close()
    for d in docs:
        d.close()


if __name__ == "__main__":
    merge_2x2(
        "cifar10_resnet18_split.pdf",
        "cifar10_resnet34_split.pdf",
        "cifar100_resnet18_split.pdf",
        "cifar100_resnet34_split.pdf",
        "merged_column.pdf",
    )
    print("done -> merged_column.pdf")
