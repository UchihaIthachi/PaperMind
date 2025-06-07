import PyPDF2
import streamlit as st # For UI feedback - to be refactored if used from non-UI layer
from io import BytesIO # To handle Streamlit UploadedFile objects more generically

def extract_text_from_pdfs(uploaded_files: list) -> str:
    """
    Extracts text content from a list of uploaded PDF files.

    Args:
        uploaded_files: A list of file-like objects (e.g., Streamlit UploadedFile)
                        representing the PDF files.

    Returns:
        A single string containing all extracted text from the PDFs.
        Returns an empty string if no files are provided or no text is extracted.
    """
    all_text = ""
    if not uploaded_files:
        return all_text

    for uploaded_file_obj in uploaded_files:
        try:
            # Streamlit's UploadedFile is already a file-like object.
            # If it's not already in a BytesIO wrapper, PyPDF2 might need one.
            # However, PyPDF2.PdfReader can often handle Streamlit's UploadedFile directly.
            # Let's ensure it's a BytesIO object for wider compatibility if direct use fails.

            # Ensure we are at the beginning of the file stream
            if hasattr(uploaded_file_obj, 'seek'):
                uploaded_file_obj.seek(0)

            reader = PyPDF2.PdfReader(uploaded_file_obj)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n" # Add newline between pages
        except Exception as e:
            # UI feedback should ideally be pushed to the UI layer.
            # For now, print error and optionally use st.error if this util is only called from Streamlit context.
            print(f"ERROR: Error reading from a PDF file ({getattr(uploaded_file_obj, 'name', 'unknown file')}): {e}")
            # Consider if st.error is appropriate here or should be handled by caller.
            # For now, keeping it as per original app.py structure. If this util is used
            # by a non-Streamlit process, st.error will fail.
            # If this function is guaranteed to be called only from Streamlit app context:
            if 'st' in globals() and hasattr(st, 'error'): # Check if streamlit is available
                 st.error(f"Error reading {getattr(uploaded_file_obj, 'name', 'a PDF file')}: {e}")
    return all_text.strip()

if __name__ == '__main__':
    print("Testing document_parsers.py...")

    # Mocking Streamlit's UploadedFile for testing
    class MockUploadedFile:
        def __init__(self, name, content_bytes):
            self.name = name
            self.file_obj = BytesIO(content_bytes)
            # PyPDF2 needs the stream to be seekable
            self.file_obj.seek(0)

        def read(self, size=-1):
            return self.file_obj.read(size)

        def seek(self, offset, whence=0):
            return self.file_obj.seek(offset, whence)

    # Create a dummy PDF in memory (very basic, may not work with all PyPDF2 features but good for text extraction test)
    # This requires reportlab or similar to create a real PDF for robust testing.
    # For now, we'll assume a simple text extraction.
    # A better test would use an actual small PDF file read into BytesIO.

    # Test with an empty list
    print("Test with empty file list:")
    text_empty = extract_text_from_pdfs([])
    print(f"Extracted text (empty list): '{text_empty}' (Expected: '')\n")

    # Test with a mock file (content is not a real PDF, so PyPDF2 might error)
    # This highlights that testing PDF extraction properly needs actual PDF byte streams.
    # For this test, we'll just check if it handles potential errors gracefully.
    print("Test with a mock non-PDF file (expecting PyPDF2 error or no text):")
    mock_file_non_pdf = MockUploadedFile("fake.pdf", b"This is not a PDF.")
    text_non_pdf = extract_text_from_pdfs([mock_file_non_pdf])
    print(f"Extracted text (non-PDF): '{text_non_pdf}' (Expected: '' or error message printed)\n")

    # To properly test, you'd need a minimal PDF file.
    # Example: Create a dummy PDF using reportlab (if available in environment)
    # from reportlab.pdfgen import canvas
    # from io import BytesIO
    # pdf_content_io = BytesIO()
    # c = canvas.Canvas(pdf_content_io)
    # c.drawString(100, 750, "Hello World from PDF.")
    # c.showPage()
    # c.save()
    # pdf_content_io.seek(0)
    # mock_pdf_file = MockUploadedFile("test.pdf", pdf_content_io.getvalue())
    # text_from_pdf = extract_text_from_pdfs([mock_pdf_file])
    # print(f"Extracted text (actual PDF): '{text_from_pdf}'")

    print("document_parsers.py test finished (proper PDF test requires PDF generation/file).")
