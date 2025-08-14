
# import torch
# import base64
# import urllib.request

# from io import BytesIO
# from PIL import Image
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# from olmocr.data.renderpdf import render_pdf_to_base64png
# from olmocr.prompts import build_finetuning_prompt
# from olmocr.prompts.anchor import get_anchor_text

# # Initialize the model
# model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Grab a sample PDF
# # urllib.request.urlretrieve("https://molmo.allenai.org/paper.pdf", "./paper.pdf")

# # Render page 1 to an image
# image_base64 = render_pdf_to_base64png("./CRL1.pdf", 1, target_longest_image_dim=1024)

# # Build the prompt, using document metadata
# anchor_text = get_anchor_text("./CRL1.pdf", 1, pdf_engine="pdfreport", target_length=4000)
# prompt = build_finetuning_prompt(anchor_text)

# # Build the full prompt
# messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
#                 ],
#             }
#         ]

# # Apply the chat template and processor
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

# inputs = processor(
#     text=[text],
#     images=[main_image],
#     padding=True,
#     return_tensors="pt",
# )
# inputs = {key: value.to(device) for (key, value) in inputs.items()}


# # Generate the output
# output = model.generate(
#             **inputs,
#             temperature=0.8,
#             max_new_tokens=50,
#             num_return_sequences=1,
#             do_sample=True,
#         )

# # Decode the output
# prompt_length = inputs["input_ids"].shape[1]
# new_tokens = output[:, prompt_length:]
# text_output = processor.tokenizer.batch_decode(
#     new_tokens, skip_special_tokens=True
# )

# print(text_output)
# # ['{"primary_language":"en","is_rotation_valid":true,"rotation_correction":0,"is_table":false,"is_diagram":false,"natural_text":"Molmo and PixMo:\\nOpen Weights and Open Data\\nfor State-of-the']


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
).eval().to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
print("========================",torch.cuda.is_available(),torch._C._cuda_getDeviceCount())
def process_pdf(pdf_path, page_num=1):
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=1024)
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text)
    return image_base64, prompt

def process_image(image_path):
    pil = Image.open(image_path)
    buffered = BytesIO()
    pil.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    prompt = "Extract text from this image.\n\n"
#     prompt = """
# Extract all visible text from this image and format it in **valid Markdown**.

# Follow these rules:
# 1. Use `#`, `##`, `###` for headings, if any are visually present.
# 2. Use bullet points `-` or numbered lists if the content is a list.
# 3. Preserve paragraph breaks with blank lines.
# 4. Do not include any extra commentary or explanation—only return valid Markdown representing the contents of the image.

# Start your output with the first heading or paragraph from the image.
# """

    return image_base64, prompt

def ocr_image_base64(image_base64, prompt):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]}
    ]
    text_template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
    inputs = processor(text=[text_template], images=[main_image], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs, temperature=0.8, max_new_tokens=10240, do_sample=True)
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return text

def to_markdown(text: str) -> str:
    # Basic formatting improvements
    lines = text.splitlines()
    md_lines = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            md_lines.append("")
        else:
            md_lines.append(f"- {ln}")
    return "\n".join(md_lines)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_md.py <path-to-pdf-or-image>")
        sys.exit(1)

    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pdf"]:
        image_base64, prompt = process_pdf(path, page_num=1)
        print("============================11")
    else:
        image_base64, prompt = process_image(path)

    text = ocr_image_base64(image_base64, prompt)
    print("============================",text)
    md = to_markdown(text)
    print("### Extracted Markdown Text:\n")
    print(md)
    output_md_path = os.path.splitext(path)[0] + ".md"
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(text.strip())

    print(f"\n✅ Markdown saved to {output_md_path}")