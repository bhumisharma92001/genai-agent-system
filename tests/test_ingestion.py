from ingestion.document_loader import DocumentLoader


loader = DocumentLoader()

chunks = loader.load(
    r"C:\Users\bhoomi.sharma\Downloads\sales_report_30_records.xlsx"
)

print("\nTOTAL CHUNKS:\n")
print(len(chunks))

print("\nFIRST 5 CHUNKS:\n")

for chunk in chunks[:5]:

    print(chunk)

    print("\n" + "=" * 100 + "\n")