import pikepdf
import re

# Original PDF open karein
pdf = pikepdf.Pdf.open('Relieving Letter (1).pdf')

for page in pdf.pages:
    # Page ka raw content bytes read karein
    stream = page.Contents.read_bytes().decode('latin-1')
    
    # 1. Name change: Animesh Tiwari -> Rishav Mishra (raw kerning format ko handle karte hue)
    stream = re.sub(r'\[\(Mr\)6\(\.\).*?\(ari\)\]', '[(Mr. Rishav Mishra)]', stream)
    
    # 2. Father's Name change: BASHISTH TIWARI -> PRAFUL CHANDRA MISHRA
    stream = re.sub(r'\[\(B\)-8\(A\).*?\(R\)-12\(I\)\]', '[(PRAFUL CHANDRA MISHRA)]', stream)
    
    # 3. Employee ID: 269 -> 287
    stream = stream.replace('(269)', '(287)')
    
    # 4. Dates Change 
    stream = stream.replace('(04th)', '(10th)')
    stream = stream.replace('(31st)', '(31st)') # Ye same rahega
    stream = stream.replace('2024', '2025') 
    
    # 5. Designation: Manager- Quality -> Quality Engineer
    stream = re.sub(r'\(Manager-.*?Quality\)', '(Quality Engineer)', stream)
    
    # 6. Bottom wala wish name change
    stream = re.sub(r'\[\(A\)16\(ni\).*?\(ari\)\]', '[(Rishav Mishra)]', stream)

    # Wapas original structure me data encode karke page me set karein
    page.Contents = pdf.make_stream(stream.encode('latin-1'))

# Naya modify kiya hua PDF save karein
pdf.save('Relieving_Letter_Final.pdf')
print("PDF successfully updated!")