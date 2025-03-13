from PyPDF2 import PdfReader
from io import BytesIO
import requests

def extract_text_from_pdf(file_url: str) -> str:
    """
    Downloads a PDF file from the given URL and extracts its text content.
    
    Args:
        file_url (str): The URL of the PDF file.
        
    Returns:
        str: The extracted text content of the PDF.
        
    Raises:
        Exception: If there is an error downloading or processing the PDF.
    """
    try:
        # Download the PDF file
        response = requests.get(file_url, stream=True, timeout=10)      # timeout in sec?
        response.raise_for_status()

        # Validate if the URL is pointing to a PDF file
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type:
            raise Exception(f"The URL does not point to a PDF file. Content-Type: {content_type}")

        # Use PyPDF2 to read the PDF content
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        combined_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                combined_text += text + "\n\n"
                
        return combined_text
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download the file: {str(e)}")
    except Exception as e:
        raise Exception(f"An error occurred while processing the PDF: {str(e)}")