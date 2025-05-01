from io import BytesIO
from xhtml2pdf import pisa

def generate_pdf_from_html(html_content: str) -> BytesIO:
    result = BytesIO()
    pisa_status = pisa.CreatePDF(html_content, dest=result)

    if pisa_status.err:
        raise Exception(f"Error al generar el PDF: {pisa_status.err}")

    pdf_data = result.getvalue()
    print(f"Tamaño del PDF generado: {len(pdf_data)} bytes")

    if len(pdf_data) == 0:
        raise Exception("El PDF generado está vacío")

    result.seek(0)
    return result