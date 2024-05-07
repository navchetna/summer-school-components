import fitz

class PDFHighlighter:
    def __init__(self, pdf_path):
        self.pdf = fitz.open(pdf_path)
    
    def highlight_text(self, text_to_highlight, output_path):
        # Iterate over each page
        for page_number in range(len(self.pdf)):
            page = self.pdf[page_number]
            
            # Search for the text in the PDF
            text_instances = page.search_for(text_to_highlight)
            
            # Highlight each instance of the text
            for inst in text_instances:
                # Create a rectangle around the text
                rect = fitz.Rect(inst.x0, inst.y0, inst.x1, inst.y1)
                
                # Highlight the rectangle
                highlight = page.add_highlight_annot(rect)
                highlight.update()
        
        # Save the highlighted PDF
        self.pdf.save(output_path, garbage=4, deflate=True)
        self.pdf.close()

