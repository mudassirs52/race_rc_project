import fitz

for fname in ['AL2002_LabProject.pdf', 'TF-IDF_Student_Manual.pdf']:
    print(f'\n\n{"="*60}')
    print(f'FILE: {fname}')
    print(f'{"="*60}')
    try:
        doc = fitz.open(fname)
        for i, page in enumerate(doc):
            print(f'\n=== PAGE {i+1} ===')
            print(page.get_text())
    except Exception as e:
        print(f'Error: {e}')
