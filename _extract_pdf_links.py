import re
from pathlib import Path

pdf_path = Path(r"g:\My Drive\MS AI\Semester_3\ATML\PAs\ATML-PA3\ATML-PA3.pdf")
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

reader = PdfReader(str(pdf_path))
print("pages", len(reader.pages))
urls = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    for m in re.findall(r"https?://[^\s\)\]\>\"'<>]+", text):
        urls.append((i + 1, m, "text"))
    annots = page.get("/Annots")
    if annots:
        for a in annots:
            obj = a.get_object()
            if obj.get("/Subtype") == "/Link" and "/A" in obj:
                act = obj["/A"]
                if "/URI" in act:
                    urls.append((i + 1, str(act["/URI"]), "annot"))

print("URLS:")
seen = set()
for page, u, kind in urls:
    key = (page, u)
    if key in seen:
        continue
    seen.add(key)
    print(f"p{page} [{kind}]: {u}")

for i in range(9, min(14, len(reader.pages))):
    t = reader.pages[i].extract_text() or ""
    idx = t.find("ATML-PA3-Data")
    print(f"\n--- page {i+1} ATML-PA3-Data idx={idx} ---")
    if idx >= 0:
        print(repr(t[max(0, idx - 120) : idx + 250]))
    # also search drive/lms/huggingface
    for needle in ("drive.google", "http", "LMS", "huggingface", "Dropbox", "github.com"):
        if needle.lower() in t.lower():
            print(f"contains {needle}")
