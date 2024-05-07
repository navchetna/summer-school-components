from highlighter import PDFHighlighter

if __name__ == "__main__":
    pdf_path = "MRF.pdf"
    text_to_highlight = """Co sold out its share in 1979 and the name of the
company was changed to MRF Ltd in the year."""
    output_path = "highlighted.pdf"
    highlighter = PDFHighlighter(pdf_path)
    highlighter.highlight_text(text_to_highlight, output_path)
    

