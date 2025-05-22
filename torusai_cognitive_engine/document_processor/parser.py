from typing import Union, IO, AnyStr
import io
import os

# Attempt to import necessary libraries, provide guidance if missing
try:
    import pdfplumber
except ImportError:
    pdfplumber = None # type: ignore

try:
    from docx import Document as DocxDocument # python-docx
except ImportError:
    DocxDocument = None # type: ignore

def extract_text_from_pdf(file_like_object: Union[str, IO[bytes]]) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_like_object: A file path string or a file-like object opened in binary read mode.

    Returns:
        The extracted text as a single string, or an empty string if extraction fails.
    """
    if pdfplumber is None:
        raise ImportError("pdfplumber library is not installed. Please install it via 'pip install pdfplumber'")
    
    text = []
    try:
        with pdfplumber.open(file_like_object) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_like_object: Union[str, IO[bytes]]) -> str:
    """
    Extracts text from a DOCX file.

    Args:
        file_like_object: A file path string or a file-like object (e.g., BytesIO).

    Returns:
        The extracted text as a single string, or an empty string if extraction fails.
    """
    if DocxDocument is None:
        raise ImportError("python-docx library is not installed. Please install it via 'pip install python-docx'")
    
    text = []
    try:
        doc = DocxDocument(file_like_object)
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_txt(file_like_object: Union[str, IO[AnyStr]]) -> str:
    """
    Extracts text from a TXT file.

    Args:
        file_like_object: A file path string or a file-like object.

    Returns:
        The extracted text as a single string, or an empty string if extraction fails.
    """
    try:
        if isinstance(file_like_object, str): # It's a path
            with open(file_like_object, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else: # It's a file-like object
            # Ensure it's in text mode or decode if binary
            if hasattr(file_like_object, 'encoding'): # TextIOWrapper
                 return file_like_object.read() # type: ignore
            else: # BytesIO or similar
                return file_like_object.read().decode('utf-8', errors='ignore') # type: ignore
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return ""

def parse_document(uploaded_file: IO[bytes], filename: str) -> str:
    """
    Parses an uploaded document (PDF, DOCX, TXT) and extracts text.

    Args:
        uploaded_file: The uploaded file object (e.g., from Streamlit's st.file_uploader).
                       This is typically a BytesIO object.
        filename: The original name of the uploaded file, used to determine type.

    Returns:
        The extracted text as a string, or an empty string if parsing fails or type is unsupported.
    """
    file_extension = os.path.splitext(filename)[1].lower()
    
    # The uploaded_file is a BytesIO object, so we pass it directly
    # For pdfplumber and python-docx, they can often handle BytesIO directly.
    # For txt, we decode it.

    if file_extension == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == ".docx":
        return extract_text_from_docx(uploaded_file)
    elif file_extension == ".txt":
        # For txt, we need to ensure it's treated as text.
        # Since uploaded_file is BytesIO, we read and decode.
        try:
            return uploaded_file.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error decoding TXT file from BytesIO: {e}")
            return ""
    else:
        print(f"Unsupported file type: {file_extension}")
        return ""

if __name__ == '__main__':
    # Example Usage (requires creating dummy files or providing paths)
    # Ensure you have dummy.pdf, dummy.docx, dummy.txt in a 'test_docs' subdir
    
    test_doc_dir = "test_docs_for_parser"
    os.makedirs(test_doc_dir, exist_ok=True)

    # Create dummy files for testing
    dummy_txt_path = os.path.join(test_doc_dir, "dummy.txt")
    with open(dummy_txt_path, "w") as f:
        f.write("This is a test text file.\nHello world.")
    
    # For PDF and DOCX, manual creation or using libraries is needed.
    # For now, we'll just test TXT here.
    # To test PDF/DOCX, you'd need to create those files and then use:
    # with open("path/to/dummy.pdf", "rb") as f_pdf:
    #     pdf_text = parse_document(f_pdf, "dummy.pdf")
    #     print("\n--- PDF Text ---")
    #     print(pdf_text)

    # with open("path/to/dummy.docx", "rb") as f_docx:
    #     docx_text = parse_document(f_docx, "dummy.docx")
    #     print("\n--- DOCX Text ---")
    #     print(docx_text)

    print(f"Testing with: {dummy_txt_path}")
    with open(dummy_txt_path, "rb") as f_txt: # Open as bytes for parse_document
        txt_text = parse_document(f_txt, "dummy.txt")
        print("\n--- TXT Text ---")
        print(txt_text)

    # Test with BytesIO (simulating Streamlit upload)
    txt_content_bytes = b"This is a BytesIO test.\nSecond line."
    bytes_io_obj = io.BytesIO(txt_content_bytes)
    txt_from_bytesio = parse_document(bytes_io_obj, "dummy_from_bytesio.txt")
    print("\n--- TXT from BytesIO ---")
    print(txt_from_bytesio)

    if pdfplumber:
        # Create a dummy PDF if possible (simplified, real PDF creation is complex)
        # This is a placeholder - proper PDF creation needs a library like reportlab
        # For a real test, have a sample.pdf file available.
        print("\nPDF parsing requires a real PDF file for testing.")
        # Example: if you have 'sample.pdf' in test_doc_dir
        sample_pdf_path = os.path.join(test_doc_dir, "sample.pdf")
        if os.path.exists(sample_pdf_path):
             with open(sample_pdf_path, "rb") as f_pdf:
                pdf_text = parse_document(f_pdf, "sample.pdf")
                print("\n--- Sample PDF Text ---")
                print(pdf_text)
        else:
            print(f"Skipping PDF test, {sample_pdf_path} not found.")

    if DocxDocument:
        print("\nDOCX parsing requires a real DOCX file for testing.")
        # Example: if you have 'sample.docx' in test_doc_dir
        sample_docx_path = os.path.join(test_doc_dir, "sample.docx")
        if os.path.exists(sample_docx_path):
            with open(sample_docx_path, "rb") as f_docx:
                docx_text = parse_document(f_docx, "sample.docx")
                print("\n--- Sample DOCX Text ---")
                print(docx_text)
        else:
            print(f"Skipping DOCX test, {sample_docx_path} not found.")