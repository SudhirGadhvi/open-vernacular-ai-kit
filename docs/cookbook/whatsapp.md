# WhatsApp Cleanup

If you have a WhatsApp export, clean it to keep only message text.

```python
from gujarati_codemix_kit import clean_whatsapp_chat_text, render_codemix

cleaned = clean_whatsapp_chat_text(raw_export_text)
out = render_codemix(cleaned, translit_mode="sentence")
```

