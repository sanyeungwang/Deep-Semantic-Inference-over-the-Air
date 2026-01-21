#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import fitz  # PyMuPDF


def merge_four_pdfs_one_row(pdf_paths, output_pdf):
    if len(pdf_paths) != 4:
        raise ValueError("need exactly 4 PDF files.")
    # open the source document and retrieve page 0.
    src_docs = [fitz.open(p) for p in pdf_paths]
    pages = [d[0] for d in src_docs]
    sizes = [(p.rect.width, p.rect.height) for p in pages]

    total_w = sum(w for w, h in sizes)  # sum of widths
    max_h = max(h for w, h in sizes)  # get the maximum height (without scaling the original page)

    out = fitz.open()
    out_page = out.new_page(width=total_w, height=max_h)

    x = 0
    for (w, h), d in zip(sizes, src_docs):
        # bottom alignment: y = max_h - h; use y=0 for top alignment, and y=(max_h - h)/2 for center alignment.
        rect = fitz.Rect(x, max_h - h, x + w, max_h)
        out_page.show_pdf_page(rect, d, 0)  # place page 0 of the source PDF into the specified rectangle
        x += w

    out.save(output_pdf)
    out.close()
    for d in src_docs:
        d.close()


if __name__ == "__main__":
    inputs = [
        "cifar100_resnet34_SP-2.pdf",
        "cifar100_resnet34_SP-3.pdf",
        "cifar100_resnet34_SP-4.pdf",
        "cifar100_resnet34_SP-5.pdf",
    ]
    merge_four_pdfs_one_row(inputs, "merged_row.pdf")
    print("done -> merged_row.pdf")
