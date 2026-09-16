"""Shared plotting helpers for synthetic experiments."""

from __future__ import annotations


def savefig_png_pdf(fig, path: str, **kwargs) -> None:
    """Save ``fig`` as PNG and a sibling PDF with the same stem."""
    fig.savefig(path, **kwargs)
    pdf_path = path.replace(".png", ".pdf") if path.endswith(".png") else f"{path}.pdf"
    fig.savefig(pdf_path, **kwargs)
