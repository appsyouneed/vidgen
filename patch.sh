#!/bin/bash
# This script patches app.py to fix the OOM issue

echo "Backing up app.py..."
cp app.py app.py.backup

echo "Applying patch..."
sed -i '/^print("Moving quantized model to GPU...")$/,/^pipes.append(pipe)$/ {
    /^print("Moving quantized model to GPU...")$/!{
        /^pipes.append(pipe)$/!d
    }
}' app.py

# Insert the new code after "Moving quantized model to GPU..."
sed -i '/^print("Moving quantized model to GPU...")$/a\
print("  - Moving text_encoder...")\
pipe.text_encoder = pipe.text_encoder.to('\''cuda'\'')\
clear_vram()\
\
print("  - Moving transformer...")\
pipe.transformer = pipe.transformer.to('\''cuda'\'')\
clear_vram()\
\
print("  - Moving transformer_2...")\
pipe.transformer_2 = pipe.transformer_2.to('\''cuda'\'')\
clear_vram()\
\
print("  - Moving VAE...")\
pipe.vae = pipe.vae.to('\''cuda'\'')\
clear_vram()\
\
# Move scheduler to GPU if it has parameters\
if hasattr(pipe, '\''scheduler'\'') and hasattr(pipe.scheduler, '\''to'\''):\
    pipe.scheduler = pipe.scheduler.to('\''cuda'\'')\
' app.py

echo "Patch applied. Original saved as app.py.backup"
echo "Run: python3 app.py"
