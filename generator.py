from io import BytesIO

from barcode import EAN13
from barcode.writer import ImageWriter

# Write to a file-like object:
rv = BytesIO()
EAN13(str(100000902922), writer=ImageWriter()).write(rv)

# Or to an actual file:
with open("barcode.jpeg", "wb") as f:
    EAN13("100000001234", writer=ImageWriter()).write(f)